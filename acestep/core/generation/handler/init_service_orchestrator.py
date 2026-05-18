"""Top-level initialization orchestration for the handler."""

import gc
import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from acestep import gpu_config
from acestep.model_downloader import DEFAULT_TURBO_DIT_MODEL

_ROCM_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _release_accelerator_cache() -> None:
    """Best-effort accelerator cache cleanup after releasing model objects."""

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    except Exception:
        pass


def _cuda_supports_bfloat16() -> bool:
    """Return whether the active CUDA device supports native bfloat16 kernels."""
    return gpu_config.cuda_supports_bfloat16()


def _resolve_rocm_dtype() -> torch.dtype:
    """Return a safe model dtype for ROCm/HIP devices.

    Uses ``float32`` by default to avoid segfaults from incomplete
    ``bfloat16`` kernel support on some ROCm GPU configurations (e.g.
    AMD iGPUs on Strix Halo).  Set the ``ACESTEP_ROCM_DTYPE`` environment
    variable to ``float16`` or ``bfloat16`` to override for hardware that
    fully supports those formats.
    """
    raw = os.environ.get("ACESTEP_ROCM_DTYPE", "float32").strip().lower()
    dtype = _ROCM_DTYPE_MAP.get(raw)
    if dtype is None:
        logger.warning(
            f"[initialize_service] Unknown ACESTEP_ROCM_DTYPE={raw!r}; "
            "falling back to float32."
        )
        dtype = torch.float32
    return dtype


class InitServiceOrchestratorMixin:
    """Public ``initialize_service`` orchestration entrypoint."""

    def _release_loaded_runtime_components(self) -> None:
        """Release the currently loaded DiT runtime before switching checkpoints."""

        had_runtime = any(
            getattr(self, attr, None) is not None
            for attr in (
                "model",
                "vae",
                "text_encoder",
                "text_tokenizer",
                "silence_latent",
                "mlx_decoder",
                "_base_decoder",
            )
        )
        if not had_runtime:
            return

        logger.info("[initialize_service] Releasing existing DiT runtime before loading new checkpoint.")
        for attr in (
            "model",
            "vae",
            "text_encoder",
            "text_tokenizer",
            "silence_latent",
            "reward_model",
            "mlx_decoder",
            "_base_decoder",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)

        self.config = None
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._active_loras = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False
        self.last_init_params = None
        gc.collect()
        _release_accelerator_cache()

    def initialize_service(
        self,
        project_root: str,
        config_path: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        use_mlx_dit: bool = True,
        vae_checkpoint: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Initialize model artifacts and runtime backends for generation.

        This method intentionally supports repeated calls to reinitialize models
        with new settings; it does not short-circuit when components are already loaded.
        """
        try:
            if config_path is None:
                config_path = DEFAULT_TURBO_DIT_MODEL
                logger.warning(
                    "[initialize_service] config_path not set; defaulting to '{}'.",
                    DEFAULT_TURBO_DIT_MODEL,
                )

            resolved_device = self._resolve_initialize_device(device)
            self.device = resolved_device
            self.offload_to_cpu = offload_to_cpu
            self.offload_dit_to_cpu = offload_dit_to_cpu

            normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
                device=resolved_device,
                compile_model=compile_model,
                quantization=quantization,
            )
            self.compiled = normalized_compile
            if resolved_device == "cuda" and gpu_config.is_rocm_available():
                self.dtype = _resolve_rocm_dtype()
                logger.info(
                    f"[initialize_service] ROCm/HIP device detected: using dtype={self.dtype} "
                    "(set ACESTEP_ROCM_DTYPE=bfloat16 or float16 to override)"
                )
            elif resolved_device == "cuda":
                if gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
                    logger.info(
                        "[initialize_service] Pre-Ampere CUDA detected: "
                        "using float16 instead of bfloat16."
                    )
            else:
                self.dtype = torch.bfloat16 if resolved_device == "xpu" else torch.float32
            self.quantization = normalized_quantization
            try:
                self._validate_quantization_setup(
                    quantization=self.quantization,
                    compile_model=normalized_compile,
                )
            except ImportError as exc:
                if self.quantization is not None:
                    logger.warning(
                        "[initialize_service] Quantization disabled: {}",
                        exc,
                    )
                    self.quantization = None
                else:
                    raise

            from acestep.model_downloader import (
                DEFAULT_VAE_VARIANT,
                get_models_dir,
            )

            checkpoint_dir = str(get_models_dir(project_root=project_root or None))
            checkpoint_path = Path(checkpoint_dir)

            # Resolve VAE selection: explicit param > env var > default.
            resolved_vae_variant = (
                vae_checkpoint
                or os.environ.get("ACESTEP_VAE_CHECKPOINT")
                or DEFAULT_VAE_VARIANT
            )

            precheck_failure = self._ensure_models_present(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                prefer_source=prefer_source,
                vae_variant=resolved_vae_variant,
            )
            if precheck_failure is not None:
                self.model = None
                self.vae = None
                self.text_encoder = None
                self.text_tokenizer = None
                self.config = None
                self.silence_latent = None
                return precheck_failure

            self._sync_model_code_if_needed(config_path, checkpoint_path)
            self._release_loaded_runtime_components()

            model_path = os.path.join(checkpoint_dir, config_path)
            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=resolved_device,
                use_flash_attention=use_flash_attention,
                compile_model=normalized_compile,
                quantization=self.quantization,
            )
            vae_path = self._load_vae_model(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
                compile_model=normalized_compile,
                vae_variant=resolved_vae_variant,
            )
            text_encoder_path = self._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
            )

            mlx_dit_status, mlx_vae_status = self._initialize_mlx_backends(
                device=resolved_device,
                use_mlx_dit=use_mlx_dit,
                mlx_compile_requested=mlx_compile_requested,
            )

            status_msg = self._build_initialize_status_message(
                device=resolved_device,
                model_path=model_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                dtype=self.dtype,
                attention=getattr(self.config, "_attn_implementation", "eager"),
                compile_model=normalized_compile,
                mlx_compile_requested=mlx_compile_requested,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                quantization=self.quantization,
                mlx_dit_status=mlx_dit_status,
                mlx_vae_status=mlx_vae_status,
            )

            self.last_init_params = {
                "project_root": project_root,
                "config_path": config_path,
                "device": resolved_device,
                "use_flash_attention": use_flash_attention,
                "compile_model": normalized_compile,
                "offload_to_cpu": offload_to_cpu,
                "offload_dit_to_cpu": offload_dit_to_cpu,
                "quantization": self.quantization,
                "use_mlx_dit": use_mlx_dit,
                "prefer_source": prefer_source,
                "vae_checkpoint": resolved_vae_variant,
            }

            return status_msg, True
        except Exception as exc:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return error_msg, False
