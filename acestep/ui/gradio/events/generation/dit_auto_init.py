"""On-demand DiT initialization helpers for generation UI actions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from acestep.model_downloader import DEFAULT_TURBO_DIT_MODEL

from .quantization import select_quantization_value


def _resolve_project_root() -> str:
    """Resolve the ACE-Step install root for local model access."""

    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if project_root:
        return str(Path(project_root).expanduser().resolve())
    return str(Path(__file__).resolve().parents[5])


def _last_init_params(dit_handler: Any) -> dict[str, Any]:
    """Return a copy of a DiT handler's last initialization parameters."""

    params = getattr(dit_handler, "last_init_params", None) or {}
    return dict(params) if isinstance(params, dict) else {}


def _selected_model(config_path: str | None) -> str:
    """Return the requested DiT model with the generation default fallback."""

    return str(config_path or "").strip() or DEFAULT_TURBO_DIT_MODEL


def _dit_needs_reinit(
    dit_handler: Any,
    *,
    config_path: str | None,
    device: str | None,
    use_flash_attention: bool,
    offload_to_cpu: bool,
    offload_dit_to_cpu: bool,
    compile_model: bool,
    quantization: str | None,
    mlx_dit: bool,
) -> bool:
    """Return whether the loaded DiT runtime differs from the requested setup."""

    if getattr(dit_handler, "model", None) is None:
        return True

    last_init_params = _last_init_params(dit_handler)
    if not last_init_params and not hasattr(dit_handler, "initialize_service"):
        return False
    if last_init_params.get("config_path") != config_path:
        return True
    if device and device != "auto" and last_init_params.get("device") != device:
        return True
    if bool(last_init_params.get("use_flash_attention")) != bool(use_flash_attention):
        return True
    if bool(last_init_params.get("offload_to_cpu")) != bool(offload_to_cpu):
        return True
    if bool(last_init_params.get("offload_dit_to_cpu")) != bool(offload_dit_to_cpu):
        return True
    if bool(last_init_params.get("compile_model")) != bool(compile_model):
        return True
    if last_init_params.get("quantization") != quantization:
        return True
    if bool(last_init_params.get("use_mlx_dit", True)) != bool(mlx_dit):
        return True
    return False


def ensure_dit_ready(
    dit_handler: Any,
    *,
    config_path: str | None = None,
    device: str | None = None,
    use_flash_attention: bool = False,
    offload_to_cpu: bool = False,
    offload_dit_to_cpu: bool = False,
    compile_model: bool = False,
    quantization: Any = None,
    mlx_dit: bool = True,
) -> tuple[bool, str]:
    """Ensure the foreground DiT runtime is ready for a generation UI action."""

    if dit_handler is None:
        return False, "DiT handler is unavailable."

    selected_model = _selected_model(config_path)
    selected_device = str(device or "auto").strip() or "auto"
    selected_quantization = select_quantization_value(
        quantization,
        device=selected_device,
    )
    if not _dit_needs_reinit(
        dit_handler,
        config_path=selected_model,
        device=selected_device,
        use_flash_attention=bool(use_flash_attention),
        offload_to_cpu=bool(offload_to_cpu),
        offload_dit_to_cpu=bool(offload_dit_to_cpu),
        compile_model=bool(compile_model),
        quantization=selected_quantization,
        mlx_dit=bool(mlx_dit),
    ):
        return True, ""

    initialize_service = getattr(dit_handler, "initialize_service", None)
    if not callable(initialize_service):
        return False, "Model not initialized. Please initialize the service first."

    logger.info("[generation] Auto-initializing DiT service: {}", selected_model)
    try:
        status, ok = initialize_service(
            project_root=_resolve_project_root(),
            config_path=selected_model,
            device=selected_device,
            use_flash_attention=bool(use_flash_attention),
            compile_model=bool(compile_model),
            offload_to_cpu=bool(offload_to_cpu),
            offload_dit_to_cpu=bool(offload_dit_to_cpu),
            quantization=selected_quantization,
            use_mlx_dit=bool(mlx_dit),
        )
    except Exception as exc:
        logger.exception("DiT auto-initialization failed")
        return False, f"Failed to auto-initialize DiT service: {exc!s}"
    if not ok:
        return False, status
    return True, f"DiT service initialized automatically.\n{status}"
