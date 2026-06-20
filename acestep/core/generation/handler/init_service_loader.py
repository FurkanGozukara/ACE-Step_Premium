"""Checkpoint and model-loading helpers for service initialization."""

from contextlib import contextmanager
import importlib
import inspect
import json
import os
from typing import Optional

import torch
from loguru import logger

from acestep import gpu_config
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
    def _build_quantization_config(quantization: str):
        """Return a torchao quantization config object for the requested mode."""
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
    ) -> None:
        """Apply torchao quantization to DiT linear layers when requested."""
        if quantization is None:
            return
        if quantization == "fp8_scaled":
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
        def _dit_filter_fn(module, fqn):
            """Keep only decoder-side DiT linear layers and exclude tokenizers."""
            if not _is_linear(module, fqn):
                return False
            parts = fqn.split(".")
            if not parts or parts[0] != "decoder":
                return False
            for part in parts:
                if part in ("tokenizer", "detokenizer"):
                    return False
            return True

        quantize_(self.model, quant_config, filter_fn=_dit_filter_fn)
        logger.info(f"[initialize_service] DiT quantized with: {quantization}")

    def _load_main_model_from_checkpoint(
        self,
        *,
        model_checkpoint_path: str,
        device: str,
        use_flash_attention: bool,
        compile_model: bool,
        quantization: Optional[str],
    ) -> str:
        """Load DiT, apply compile/quantization options, and return selected attention backend."""
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
        direct_device_load = self._should_direct_load_main_model(device)
        if use_transformers_meta_init_compat and direct_device_load:
            logger.info(
                "[initialize_service] Transformers meta-init compatibility active for "
                "ACE-Step checkpoint; preserving direct CUDA load with FSQ metadata "
                "compatibility."
            )
        loaded_directly = False
        for candidate in attn_candidates:
            direct_attempts = [True, False] if direct_device_load else [False]
            for attempt_direct in direct_attempts:
                try:
                    logger.info(
                        "[initialize_service] Attempting to load model with attention "
                        "implementation: {} direct_device_load={}",
                        candidate,
                        attempt_direct,
                    )
                    load_kwargs = self._main_model_from_pretrained_kwargs(
                        candidate,
                        device=device,
                        direct_device_load=attempt_direct,
                    )
                    with self._transformers_meta_init_compatibility_context(
                        enabled=use_transformers_meta_init_compat,
                        direct_device_load=attempt_direct,
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
            logger.info("[initialize_service] Main model loaded directly on {}", device)
        elif not self.offload_to_cpu:
            self.model = self.model.to(device).to(self.dtype)
        elif not self.offload_dit_to_cpu:
            logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
            self.model = self.model.to(device).to(self.dtype)
        else:
            self.model = self.model.to("cpu").to(self.dtype)
        self.model.eval()

        decoder = getattr(self.model, "decoder", self.model)
        if compile_model:
            self._ensure_len_for_compile(decoder, "model.decoder")
        compile_result = compile_module_forward(
            decoder,
            label="ACE-Step DiT decoder",
            enabled=compile_model,
        )
        if compile_model:
            if not compile_result.compiled:
                logger.warning(
                    "[initialize_service] torch.compile disabled for DiT decoder: {}",
                    compile_result.detail,
                )
        self._apply_dit_quantization(
            quantization,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        )

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

    def _should_direct_load_main_model(self, device: str) -> bool:
        """Return whether the main model's final placement is CUDA."""

        final_device = "cpu" if self.offload_to_cpu and self.offload_dit_to_cpu else device
        return str(final_device).startswith("cuda")
