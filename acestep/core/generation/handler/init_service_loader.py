"""Checkpoint and model-loading helpers for service initialization."""

from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import os
import threading
import time
from typing import Any, Optional

import torch
from loguru import logger

from acestep import gpu_config
from acestep.torch_compile_policy import resolve_inference_compile_policy
from acestep.torch_compile_runtime import compile_module_forward
from .fp8_scaled_quantization import apply_fp8_scaled_quantization
from .init_service_loader_components import InitServiceLoaderComponentsMixin


class InitServiceLoaderMixin(InitServiceLoaderComponentsMixin):
    """Helpers for heavy model component loading."""

    def _cuda_supports_bool_argsort(self) -> bool:
        """Return whether CUDA argsort supports bool tensors on the active device."""
        if not torch.cuda.is_available():
            return True
        target_device = str(getattr(self, "device", "cuda"))
        if not target_device.startswith("cuda"):
            target_device = "cuda"
        try:
            mask_cat = torch.tensor([[True, False]], device=target_device)
            _ = mask_cat.argsort(dim=1, descending=True, stable=True)
            return True
        except RuntimeError as exc:
            logger.debug(
                "[_cuda_supports_bool_argsort] Treating CUDA bool argsort probe failure as unsupported: {}",
                exc,
            )
            return False

    def _apply_cuda_bool_argsort_workaround(self) -> None:
        """Patch dynamic model helpers when bool argsort is unsupported on CUDA."""
        target_device = str(getattr(self, "device", ""))
        if not target_device.startswith("cuda"):
            return
        if self._cuda_supports_bool_argsort():
            return

        model_module_name = getattr(self.model.__class__, "__module__", "")
        if not model_module_name:
            return

        try:
            model_module = importlib.import_module(model_module_name)
        except Exception as exc:
            logger.warning(
                "[initialize_service] Failed to import model module for CUDA bool-argsort workaround: {}",
                exc,
            )
            return

        original_pack_sequences = getattr(model_module, "pack_sequences", None)
        if original_pack_sequences is None:
            return
        if getattr(original_pack_sequences, "__acestep_bool_argsort_patched__", False):
            return

        def _pack_sequences_cuda_compat(hidden1, hidden2, mask1, mask2):
            # ``pack_sequences`` only needs sortable integer-like masks here; keep
            # truthy/falsey semantics while avoiding CUDA bool argsort failures.
            if isinstance(mask1, torch.Tensor) and mask1.is_cuda and mask1.dtype == torch.bool:
                mask1 = mask1.to(torch.int32)
            if isinstance(mask2, torch.Tensor) and mask2.is_cuda and mask2.dtype == torch.bool:
                mask2 = mask2.to(torch.int32)
            return original_pack_sequences(hidden1, hidden2, mask1, mask2)

        _pack_sequences_cuda_compat.__acestep_bool_argsort_patched__ = True
        setattr(model_module, "pack_sequences", _pack_sequences_cuda_compat)
        logger.warning(
            "[initialize_service] Applied CUDA bool-argsort workaround to {}.pack_sequences",
            model_module_name,
        )

    @staticmethod
    def _format_gib(num_bytes: int | float | None) -> str:
        """Format a byte count as GiB for human-readable startup logs."""
        if num_bytes is None:
            return "n/a"
        return f"{float(num_bytes) / (1024 ** 3):.2f}GiB"

    def _runtime_telemetry_details(self, device: str | None = None) -> str:
        """Return process and CUDA telemetry for long initialization logs."""
        details: list[str] = []
        try:
            import psutil

            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            detail_parts = [
                f"rss={self._format_gib(getattr(mem, 'rss', None))}",
                f"vms={self._format_gib(getattr(mem, 'vms', None))}",
                f"cpu={sum(process.cpu_times()[:2]):.1f}s",
            ]
            try:
                full_mem = process.memory_full_info()
                private = getattr(full_mem, "private", None) or getattr(full_mem, "uss", None)
                if private is not None:
                    detail_parts.append(f"private={self._format_gib(private)}")
            except Exception:
                pass
            try:
                io = process.io_counters()
                detail_parts.append(f"read={self._format_gib(getattr(io, 'read_bytes', None))}")
                detail_parts.append(f"write={self._format_gib(getattr(io, 'write_bytes', None))}")
            except Exception:
                pass
            details.append("proc[" + ", ".join(detail_parts) + "]")
        except Exception:
            details.append("proc[n/a]")

        if torch.cuda.is_available():
            try:
                requested_device = str(device or getattr(self, "device", "cuda"))
                if not requested_device.startswith("cuda"):
                    requested_device = "cuda"
                cuda_device = torch.device(requested_device)
                index = cuda_device.index
                if index is None:
                    index = torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                allocated = torch.cuda.memory_allocated(index)
                reserved = torch.cuda.memory_reserved(index)
                details.append(
                    "cuda{}[free={}, total={}, torch_alloc={}, torch_reserved={}]".format(
                        index,
                        self._format_gib(free_bytes),
                        self._format_gib(total_bytes),
                        self._format_gib(allocated),
                        self._format_gib(reserved),
                    )
                )
            except Exception as exc:
                details.append(f"cuda[n/a: {exc}]")
        return " ".join(details)

    @contextmanager
    def _logged_long_operation(
        self,
        label: str,
        *,
        interval_seconds: int = 15,
        telemetry_device: str | None = None,
    ):
        """Log a heartbeat while a blocking model-load operation is running."""
        start = time.monotonic()
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(interval_seconds):
                elapsed = time.monotonic() - start
                logger.info(
                    "[initialize_service] {} still running ({:.0f}s elapsed; {})",
                    label,
                    elapsed,
                    self._runtime_telemetry_details(telemetry_device),
                )

        logger.info(
            "[initialize_service] {} started ({})",
            label,
            self._runtime_telemetry_details(telemetry_device),
        )
        thread = threading.Thread(
            target=_heartbeat,
            name="acestep-init-progress",
            daemon=True,
        )
        thread.start()
        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            stop_event.set()
            elapsed = time.monotonic() - start
            if failed:
                logger.info(
                    "[initialize_service] {} failed after {:.1f}s ({})",
                    label,
                    elapsed,
                    self._runtime_telemetry_details(telemetry_device),
                )
            else:
                logger.info(
                    "[initialize_service] {} finished in {:.1f}s ({})",
                    label,
                    elapsed,
                    self._runtime_telemetry_details(telemetry_device),
                )

    @staticmethod
    def _dit_quantization_filter(module, fqn: str, is_linear_fn) -> bool:
        """Return whether a module should receive DiT weight-only quantization."""
        if not is_linear_fn(module, fqn):
            return False
        parts = fqn.split(".")
        if not parts or parts[0] != "decoder":
            return False
        for part in parts:
            if part in ("tokenizer", "detokenizer"):
                return False
        return True

    def _dit_quantization_target_names(self, is_linear_fn) -> list[str]:
        """Return decoder linear module names targeted by DiT quantization."""
        named_modules = getattr(self.model, "named_modules", None)
        if not callable(named_modules):
            return []
        return [
            name
            for name, module in named_modules()
            if self._dit_quantization_filter(module, name, is_linear_fn)
        ]

    def _dit_quantization_cache_info(
        self,
        *,
        quantization: str,
        model_checkpoint_path: str | None,
        device: str,
        attn_implementation: str | None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Return cache path and metadata for a quantized DiT warm cache."""
        if not model_checkpoint_path:
            return None
        if os.environ.get("ACESTEP_DISABLE_DIT_QUANT_CACHE", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            logger.info("[initialize_service] Quantized DiT cache disabled by environment.")
            return None

        checkpoint_path = os.path.abspath(model_checkpoint_path)
        metadata = {
            "format": "acestep_dit_quant_cache_v1",
            "checkpoint": checkpoint_path,
            "checkpoint_files": self._checkpoint_cache_fingerprint(checkpoint_path),
            "quantization": quantization,
            "dtype": str(self.dtype),
            "device": device,
            "attention": attn_implementation or "",
            "offload_to_cpu": bool(self.offload_to_cpu),
            "offload_dit_to_cpu": bool(self.offload_dit_to_cpu),
            "torch": getattr(torch, "__version__", "unknown"),
            "torchao": self._torchao_version(),
            "filter": "decoder_linear_no_tokenizers_v1",
        }
        cache_key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        cache_root = os.path.join(
            self._get_project_root(),
            ".cache",
            "acestep",
            "quantized_dit",
        )
        return os.path.join(cache_root, f"{cache_key}.pt"), metadata

    @staticmethod
    def _torchao_version() -> str:
        """Return the installed TorchAO version label for cache invalidation."""
        try:
            import torchao
        except Exception:
            return "unavailable"
        return str(getattr(torchao, "__version__", "unknown"))

    @staticmethod
    def _checkpoint_cache_fingerprint(checkpoint_path: str) -> list[dict[str, Any]]:
        """Return lightweight file metadata used to invalidate stale caches."""
        if not os.path.isdir(checkpoint_path):
            return []
        relevant_suffixes = (".bin", ".json", ".py", ".safetensors")
        files: list[dict[str, Any]] = []
        for root, dirnames, filenames in os.walk(checkpoint_path):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in sorted(filenames):
                if not filename.endswith(relevant_suffixes):
                    continue
                full_path = os.path.join(root, filename)
                try:
                    stat = os.stat(full_path)
                except OSError:
                    continue
                files.append(
                    {
                        "path": os.path.relpath(full_path, checkpoint_path).replace("\\", "/"),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        return files

    def _checkpoint_file_summary(self, checkpoint_path: str) -> str:
        """Return a compact checkpoint file summary for startup diagnostics."""
        if not os.path.isdir(checkpoint_path):
            return "missing"
        suffixes = (".bin", ".json", ".py", ".pt", ".safetensors")
        files: list[tuple[str, int]] = []
        for root, dirnames, filenames in os.walk(checkpoint_path):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in filenames:
                if not filename.endswith(suffixes):
                    continue
                full_path = os.path.join(root, filename)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                rel_path = os.path.relpath(full_path, checkpoint_path).replace("\\", "/")
                files.append((rel_path, size))
        files.sort(key=lambda item: item[1], reverse=True)
        largest = ", ".join(
            f"{name}={self._format_gib(size)}" for name, size in files[:3]
        )
        return (
            f"files={len(files)}, total={self._format_gib(sum(size for _, size in files))}, "
            f"largest=[{largest or 'none'}]"
        )

    def _restore_dit_quantization_cache(
        self,
        *,
        cache_path: str,
        metadata: dict[str, Any],
        target_module_names: list[str],
        quantization: str,
    ) -> bool:
        """Restore cached TorchAO decoder weights into a freshly loaded model."""
        if not os.path.exists(cache_path):
            logger.info(
                "[initialize_service] Quantized DiT cache miss: {}",
                cache_path,
            )
            return False
        try:
            with self._logged_long_operation(
                f"Stage 3/4: Loading quantized DiT cache from {cache_path}",
                interval_seconds=10,
                telemetry_device=str(getattr(self, "device", "")) or None,
            ):
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("metadata") != metadata:
                logger.warning(
                    "[initialize_service] Ignoring stale quantized DiT cache metadata: {}",
                    cache_path,
                )
                return False
            cached_weights = payload.get("weights")
            if not isinstance(cached_weights, dict):
                raise ValueError("cache payload is missing weights")
            missing = [name for name in target_module_names if name not in cached_weights]
            if missing:
                raise ValueError(f"cache payload is missing {len(missing)} decoder weights")

            modules = dict(self.model.named_modules())
            for name in target_module_names:
                module = modules[name]
                weight = cached_weights[name]
                if not isinstance(weight, torch.Tensor):
                    raise TypeError(f"cached weight for {name} is not a tensor")
                current_weight = getattr(module, "weight", None)
                target_device = (
                    current_weight.device
                    if isinstance(current_weight, torch.Tensor)
                    else torch.device("cpu")
                )
                module._parameters["weight"] = torch.nn.Parameter(
                    weight.to(target_device),
                    requires_grad=False,
                )
            logger.info(
                "[initialize_service] Restored {} {} decoder weights from quantized DiT cache.",
                len(target_module_names),
                quantization,
            )
            return True
        except Exception as exc:
            logger.warning(
                "[initialize_service] Failed to restore quantized DiT cache {}; "
                "rebuilding once. Error: {}",
                cache_path,
                exc,
            )
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return False

    def _save_dit_quantization_cache(
        self,
        *,
        cache_path: str,
        metadata: dict[str, Any],
        target_module_names: list[str],
        quantization: str,
    ) -> None:
        """Persist quantized decoder weights for the next CPU-offloaded startup."""
        if not target_module_names:
            return
        tmp_path: str | None = None
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            modules = dict(self.model.named_modules())
            weights: dict[str, torch.Tensor] = {}
            for name in target_module_names:
                weight = getattr(modules[name], "weight", None)
                if not isinstance(weight, torch.Tensor):
                    raise TypeError(f"quantized weight for {name} is not a tensor")
                weights[name] = weight.detach().cpu()

            tmp_path = f"{cache_path}.{os.getpid()}.tmp"
            payload = {"metadata": metadata, "weights": weights}
            with self._logged_long_operation(
                f"Stage 4/4: Saving {quantization} DiT cache to {cache_path}",
                interval_seconds=10,
                telemetry_device=str(getattr(self, "device", "")) or None,
            ):
                torch.save(payload, tmp_path)
                os.replace(tmp_path, cache_path)
            size_gb = os.path.getsize(cache_path) / (1024 ** 3)
            logger.info(
                "[initialize_service] {} DiT cache ready: {} ({:.2f} GiB).",
                quantization,
                cache_path,
                size_gb,
            )
        except Exception as exc:
            logger.warning(
                "[initialize_service] Could not save quantized DiT cache {}; continuing. Error: {}",
                cache_path,
                exc,
            )
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def _build_quantization_config(quantization: str):
        """Return a torchao quantization config object for the requested mode."""
        if quantization == "int4_weight_only":
            from torchao.quantization import IntxWeightOnlyConfig
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_primitives import MappingType
            return IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(32),
                mapping_type=MappingType.SYMMETRIC,
            )
        if quantization == "int8_weight_only":
            from torchao.quantization import Int8WeightOnlyConfig
            return Int8WeightOnlyConfig()
        if quantization == "fp8_weight_only":
            from torchao.quantization import Float8WeightOnlyConfig
            return Float8WeightOnlyConfig()
        if quantization == "w8a8_dynamic":
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
            return Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
        raise ValueError(f"Unsupported quantization type: {quantization}")

    def _apply_dit_quantization(
        self,
        quantization: Optional[str],
        *,
        model_checkpoint_path: str | None = None,
        device: str = "auto",
        attn_implementation: str | None = None,
    ) -> None:
        """Apply torchao quantization to DiT linear layers when requested."""
        if quantization is None:
            return
        if quantization == "fp8_scaled":
            with self._logged_long_operation(
                "Applying scaled FP8 DiT quantization cache",
                interval_seconds=10,
                telemetry_device=device,
            ):
                apply_fp8_scaled_quantization(
                    self.model,
                    checkpoint_path=model_checkpoint_path,
                    device=device,
                )
            logger.info("[initialize_service] DiT quantized with: fp8_scaled")
            return
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import _is_linear

        quant_config = self._build_quantization_config(quantization)

        target_module_names = self._dit_quantization_target_names(_is_linear)
        if target_module_names:
            logger.info(
                "[initialize_service] DiT quantization targets: {} decoder linear layers.",
                len(target_module_names),
            )
        target_module_set = set(target_module_names)

        progress_step = max(1, min(25, len(target_module_names) // 10 or 1))
        progress_seen: set[str] = set()

        def _dit_filter_fn(module, fqn):
            """Keep decoder-side DiT linear layers and log conversion progress."""
            selected = self._dit_quantization_filter(module, fqn, _is_linear)
            if selected and fqn in target_module_set and fqn not in progress_seen:
                progress_seen.add(fqn)
                current = len(progress_seen)
                total = len(target_module_names)
                if current == 1 or current == total or current % progress_step == 0:
                    logger.info(
                        "[initialize_service] Quantizing {} DiT layer {}/{}: {}",
                        quantization,
                        current,
                        total,
                        fqn,
                    )
            return selected

        cache_info = self._dit_quantization_cache_info(
            quantization=quantization,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
            attn_implementation=attn_implementation,
        )
        if cache_info is not None and target_module_names:
            cache_path, cache_metadata = cache_info
            if self._restore_dit_quantization_cache(
                cache_path=cache_path,
                metadata=cache_metadata,
                target_module_names=target_module_names,
                quantization=quantization,
            ):
                logger.info(f"[initialize_service] DiT quantized with: {quantization} (cache)")
                return
        else:
            cache_path = None
            cache_metadata = None

        with self._logged_long_operation(
            f"Stage 3/4: First-time {quantization} DiT quantization",
            interval_seconds=10,
            telemetry_device=device,
        ):
            quantize_(self.model, quant_config, filter_fn=_dit_filter_fn)
        logger.info(f"[initialize_service] DiT quantized with: {quantization}")
        if cache_path and cache_metadata and target_module_names:
            self._save_dit_quantization_cache(
                cache_path=cache_path,
                metadata=cache_metadata,
                target_module_names=target_module_names,
                quantization=quantization,
            )

    def _load_main_model_from_checkpoint(
        self,
        *,
        model_checkpoint_path: str,
        device: str,
        use_flash_attention: bool,
        compile_model: bool,
        quantization: Optional[str],
    ) -> str:
        """Load DiT, apply inference policy/quantization, and select attention."""
        from transformers import AutoModel

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"ACE-Step V1.5 checkpoint not found at {model_checkpoint_path}")

        if torch.cuda.is_available():
            if getattr(self, "model", None) is not None:
                del self.model
                self.model = None
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                logger.warning(
                    "[initialize_service] cuda.synchronize() failed during pre-load cleanup: {}. "
                    "Continuing with fresh load attempt.",
                    exc,
                )

        if use_flash_attention and self.is_flash_attention_available(device):
            attn_implementation = "flash_attention_2"
        elif device == "cuda" and not gpu_config.cuda_supports_bfloat16():
            # Pre-Ampere GPUs (compute capability < 8.0) run in float16 which
            # can overflow in SDPA's fused softmax with longer sequences,
            # producing NaN/Inf latents (see issues #924, #927).  Eager
            # attention upcasts to float32 for softmax, avoiding the overflow.
            logger.info(
                "[initialize_service] Pre-Ampere CUDA detected: using eager "
                "attention for float16 numerical stability."
            )
            attn_implementation = "eager"
        else:
            if use_flash_attention:
                logger.warning(
                    f"[initialize_service] Flash attention requested but unavailable for device={device}. "
                    "Falling back to SDPA."
                )
            attn_implementation = "sdpa"

        attn_candidates = [attn_implementation]
        if "sdpa" not in attn_candidates:
            attn_candidates.append("sdpa")
        if "eager" not in attn_candidates:
            attn_candidates.append("eager")

        last_attn_error = None
        self.model = None
        use_transformers_meta_init_compat = self._should_disable_transformers_meta_init(
            model_checkpoint_path
        )
        load_device = self._main_model_load_device(device)
        direct_device_load = self._should_direct_load_main_model(
            load_device,
            quantization=quantization,
        )
        if use_transformers_meta_init_compat and direct_device_load:
            logger.info(
                "[initialize_service] Transformers meta-init compatibility active for "
                "ACE-Step checkpoint; preserving streamed device load with FSQ metadata "
                "compatibility."
            )
        logger.info(
            "[initialize_service] DiT load plan: checkpoint={} ({}) device={} dtype={} "
            "quantization={} compile_model={} offload_to_cpu={} offload_dit_to_cpu={} "
            "load_device={} direct_device_load={} meta_init_compat={} "
            "attention_candidates={} runtime={}",
            model_checkpoint_path,
            self._checkpoint_file_summary(model_checkpoint_path),
            device,
            self.dtype,
            quantization or "disabled",
            compile_model,
            self.offload_to_cpu,
            self.offload_dit_to_cpu,
            load_device,
            direct_device_load,
            use_transformers_meta_init_compat,
            attn_candidates,
            self._runtime_telemetry_details(device),
        )
        loaded_directly = False
        for candidate in attn_candidates:
            direct_attempts = [True, False] if direct_device_load else [False]
            for attempt_direct in direct_attempts:
                try:
                    logger.info(
                        "[initialize_service] Attempting to load model with attention "
                        "implementation: {} load_device={} direct_device_load={}",
                        candidate,
                        load_device,
                        attempt_direct,
                    )
                    load_kwargs = self._main_model_from_pretrained_kwargs(
                        candidate,
                        device=load_device,
                        direct_device_load=attempt_direct,
                    )
                    with self._transformers_meta_init_compatibility_context(
                        enabled=use_transformers_meta_init_compat,
                        direct_device_load=attempt_direct,
                    ):
                        with self._logged_long_operation(
                            "Stage 1/4: Loading DiT checkpoint before quantization "
                            f"({candidate}, load_device={load_device}, "
                            f"direct_device_load={attempt_direct})",
                            interval_seconds=15,
                            telemetry_device=device,
                        ):
                            self.model = AutoModel.from_pretrained(
                                model_checkpoint_path,
                                **load_kwargs,
                            )
                    loaded_directly = attempt_direct
                    attn_implementation = candidate
                    break
                except Exception as exc:
                    last_attn_error = exc
                    if attempt_direct:
                        logger.warning(
                            "[initialize_service] Direct device load failed with {}: {}. "
                            "Retrying staged load.",
                            candidate,
                            exc,
                        )
                    else:
                        logger.warning(
                            "[initialize_service] Failed to load model with {}: {}",
                            candidate,
                            exc,
                        )
            if self.model is not None:
                break

        if self.model is None:
            raise RuntimeError(
                f"Failed to load model with attention implementations {attn_candidates}: {last_attn_error}"
            ) from last_attn_error

        self.model.config._attn_implementation = attn_implementation
        self.config = self.model.config
        self._sync_alignment_config()
        self._apply_cuda_bool_argsort_workaround()

        if loaded_directly:
            logger.info("[initialize_service] Main model loaded directly on {}", load_device)
        elif not self.offload_to_cpu:
            with self._logged_long_operation(
                f"Stage 2/4: Moving DiT model to {device} with dtype {self.dtype}",
                interval_seconds=15,
                telemetry_device=device,
            ):
                self.model = self.model.to(device).to(self.dtype)
        elif not self.offload_dit_to_cpu:
            logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
            with self._logged_long_operation(
                f"Stage 2/4: Moving DiT model to {device} with dtype {self.dtype}",
                interval_seconds=15,
                telemetry_device=device,
            ):
                self.model = self.model.to(device).to(self.dtype)
        else:
            with self._logged_long_operation(
                f"Stage 2/4: Moving DiT model to cpu with dtype {self.dtype}",
                interval_seconds=15,
                telemetry_device=device,
            ):
                self.model = self.model.to("cpu").to(self.dtype)
        self.model.eval()

        decoder = getattr(self.model, "decoder", self.model)
        compile_policy = resolve_inference_compile_policy(compile_model)
        if compile_policy.dit:
            self._ensure_len_for_compile(decoder, "model.decoder")
        compile_result = compile_module_forward(
            decoder,
            label="ACE-Step DiT decoder",
            enabled=compile_policy.dit,
            disabled_detail=compile_policy.disabled_detail("dit"),
        )
        if compile_model and compile_policy.dit and not compile_result.compiled:
            logger.warning(
                "[initialize_service] torch.compile unavailable for DiT decoder: {}",
                compile_result.detail,
            )
        elif compile_model and not compile_policy.dit:
            logger.info(
                "[initialize_service] DiT decoder remains eager: {}",
                compile_result.detail,
            )
        self._apply_dit_quantization(
            quantization,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
            attn_implementation=attn_implementation,
        )
        if (
            loaded_directly
            and self.offload_to_cpu
            and self.offload_dit_to_cpu
            and str(load_device) != "cpu"
        ):
            with self._logged_long_operation(
                "Stage 4/4: Offloading quantized DiT model to CPU",
                interval_seconds=15,
                telemetry_device=device,
            ):
                self.model = self.model.to("cpu")

        silence_latent_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
        if not os.path.exists(silence_latent_path):
            raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
        self.silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
        silence_latent_device = "cpu" if self.offload_to_cpu and self.offload_dit_to_cpu else device
        self.silence_latent = self.silence_latent.to(silence_latent_device).to(self.dtype)
        return attn_implementation

    def _main_model_from_pretrained_kwargs(
        self,
        attn_implementation: str,
        *,
        device: str,
        direct_device_load: bool,
    ) -> dict:
        """Return ``AutoModel.from_pretrained`` kwargs for the main model."""

        kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
            "dtype": self.dtype,
        }
        if direct_device_load:
            kwargs["device_map"] = {"": device}
            kwargs["low_cpu_mem_usage"] = True
        return kwargs

    def _main_model_load_device(self, device: str) -> str:
        """Return the checkpoint streaming target for the configured DiT placement."""

        if self.offload_to_cpu and self.offload_dit_to_cpu:
            return "cpu"
        return device

    @staticmethod
    def _is_acestep_transformers_checkpoint(model_checkpoint_path: str) -> bool:
        """Return whether a checkpoint config points to ACE-Step remote code."""

        config_file = os.path.join(model_checkpoint_path, "config.json")
        try:
            with open(config_file, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False

        if str(config.get("model_type", "")).lower() == "acestep":
            return True

        architectures = config.get("architectures") or []
        auto_map = config.get("auto_map") or {}
        markers = [*architectures, *auto_map.values()]
        return any("acestep" in str(marker).lower() for marker in markers)

    @staticmethod
    def _is_transformers_meta_device_context(context_manager) -> bool:
        """Return whether a Transformers init context enters the meta device."""

        return isinstance(context_manager, torch.device) and context_manager.type == "meta"

    @classmethod
    def _transformers_uses_meta_init_for_pretrained_models(cls) -> bool:
        """Return whether installed Transformers builds pretrained models on meta."""

        try:
            from transformers.modeling_utils import PreTrainedModel
        except Exception:
            return False

        descriptor = PreTrainedModel.__dict__.get("get_init_context")
        original_func = getattr(descriptor, "__func__", None)
        if original_func is None:
            return False
        try:
            init_context_args = [PreTrainedModel, torch.float32, False, False]
            if "allow_all_kernels" in inspect.signature(original_func).parameters:
                init_context_args.append(None)
            contexts = original_func(*init_context_args)
        except Exception:
            return False
        return any(cls._is_transformers_meta_device_context(context) for context in contexts)

    def _should_disable_transformers_meta_init(self, model_checkpoint_path: str) -> bool:
        """Return whether ACE-Step needs to opt out of Transformers meta init."""

        return (
            self._is_acestep_transformers_checkpoint(model_checkpoint_path)
            and self._transformers_uses_meta_init_for_pretrained_models()
        )

    @staticmethod
    def _module_has_meta_tensors(module) -> bool:
        """Return whether a module still contains meta parameters or buffers."""

        for tensor in module.parameters(recurse=True):
            if getattr(tensor, "is_meta", False):
                return True
        for tensor in module.buffers(recurse=True):
            if getattr(tensor, "is_meta", False):
                return True
        return False

    @contextmanager
    def _transformers_meta_init_compatibility_context(
        self,
        *,
        enabled: bool,
        direct_device_load: bool = False,
    ):
        """Keep ACE-Step compatible with Transformers 5 meta construction.

        ACE-Step remote code builds ResidualFSQ during ``__init__``.  That
        dependency performs scalar checks and creates computed non-persistent
        buffers, both of which are invalid or lossy under a pure meta-device
        construction path.  For direct device loads we keep Transformers' fast
        meta initialization for checkpoint-backed weights and force only the
        tiny FSQ metadata tensors onto CPU.  Staged fallback loads use the older
        broad opt-out path.
        """

        if not enabled:
            yield
            return

        try:
            from transformers.modeling_utils import PreTrainedModel
        except Exception:
            yield
            return

        get_init_context_descriptor = PreTrainedModel.__dict__.get("get_init_context")
        get_init_context_func = getattr(get_init_context_descriptor, "__func__", None)
        move_missing_func = PreTrainedModel.__dict__.get(
            "_move_missing_keys_from_meta_to_device"
        )
        if get_init_context_func is None or not callable(move_missing_func):
            yield
            return

        if direct_device_load:
            with self._transformers_meta_direct_load_context(
                move_missing_func=move_missing_func
            ):
                yield
            return

        def _acestep_get_init_context(cls, *args, **kwargs):
            contexts = get_init_context_func(cls, *args, **kwargs)
            return [
                context
                for context in contexts
                if not self._is_transformers_meta_device_context(context)
            ]

        def _acestep_move_missing_keys_from_meta_to_device(
            model,
            *args,
            **kwargs,
        ):
            if not self._module_has_meta_tensors(model):
                return None
            return move_missing_func(model, *args, **kwargs)

        PreTrainedModel.get_init_context = classmethod(_acestep_get_init_context)
        PreTrainedModel._move_missing_keys_from_meta_to_device = (
            _acestep_move_missing_keys_from_meta_to_device
        )
        try:
            yield
        finally:
            PreTrainedModel.get_init_context = get_init_context_descriptor
            PreTrainedModel._move_missing_keys_from_meta_to_device = move_missing_func

    @contextmanager
    def _transformers_meta_direct_load_context(self, *, move_missing_func):
        """Patch only ACE-Step FSQ metadata while keeping fast meta init enabled."""

        try:
            from transformers.integrations.accelerate import get_device
            from transformers.modeling_utils import (
                PreTrainedModel,
                _load_parameter_into_model,
            )
            from vector_quantize_pytorch import finite_scalar_quantization as fsq_module
            from vector_quantize_pytorch import residual_fsq as residual_fsq_module
        except Exception:
            yield
            return

        original_residual_tensor = getattr(residual_fsq_module, "tensor", None)
        original_fsq_tensor = getattr(fsq_module, "tensor", None)
        original_torch_arange = torch.arange

        def _cpu_tensor_when_default_device_is_meta(original_func):
            def _wrapped_tensor(*args, **kwargs):
                if (
                    kwargs.get("device") is None
                    and torch.get_default_device().type == "meta"
                ):
                    kwargs = dict(kwargs)
                    kwargs["device"] = "cpu"
                return original_func(*args, **kwargs)

            return _wrapped_tensor

        def _cpu_arange_when_default_device_is_meta(*args, **kwargs):
            if (
                kwargs.get("device") is None
                and torch.get_default_device().type == "meta"
            ):
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return original_torch_arange(*args, **kwargs)

        def _acestep_move_missing_keys_from_meta_to_device(
            model,
            missing_keys,
            device_map,
            device_mesh,
            hf_quantizer,
        ):
            if device_mesh is not None:
                return move_missing_func(
                    model,
                    missing_keys,
                    device_map,
                    device_mesh,
                    hf_quantizer,
                )

            tied_keys = getattr(model, "all_tied_weights_keys", {})
            if hasattr(tied_keys, "keys"):
                tied_keys = set(tied_keys.keys())
            else:
                tied_keys = set(tied_keys or [])

            for key in set(missing_keys) - tied_keys:
                param = model.get_parameter_or_buffer(key)
                param_device = get_device(device_map, key, valid_torch_device=True)
                if getattr(param, "is_meta", False):
                    value = torch.empty_like(param, device=param_device)
                else:
                    value = param.to(param_device)
                _load_parameter_into_model(model, key, value)

            for key, buffer in model.named_non_persistent_buffers():
                buffer_device = get_device(device_map, key, valid_torch_device=True)
                if getattr(buffer, "is_meta", False):
                    value = torch.empty_like(buffer, device=buffer_device)
                else:
                    value = buffer.to(buffer_device)
                _load_parameter_into_model(model, key, value)
            return None

        if original_residual_tensor is not None:
            residual_fsq_module.tensor = _cpu_tensor_when_default_device_is_meta(
                original_residual_tensor
            )
        if original_fsq_tensor is not None:
            fsq_module.tensor = _cpu_tensor_when_default_device_is_meta(
                original_fsq_tensor
            )
        torch.arange = _cpu_arange_when_default_device_is_meta
        PreTrainedModel._move_missing_keys_from_meta_to_device = (
            _acestep_move_missing_keys_from_meta_to_device
        )
        try:
            yield
        finally:
            if original_residual_tensor is not None:
                residual_fsq_module.tensor = original_residual_tensor
            if original_fsq_tensor is not None:
                fsq_module.tensor = original_fsq_tensor
            torch.arange = original_torch_arange
            PreTrainedModel._move_missing_keys_from_meta_to_device = move_missing_func

    def _should_direct_load_main_model(
        self,
        device: str,
        *,
        quantization: Optional[str] = None,
    ) -> bool:
        """Return whether Transformers should stream weights through a device map."""

        device_text = str(device)
        return device_text == "cpu" or device_text.startswith("cuda")
