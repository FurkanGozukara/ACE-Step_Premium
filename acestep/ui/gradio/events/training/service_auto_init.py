"""On-demand service initialization for training dataset actions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from acestep.gpu_config import get_global_gpu_config
from acestep.model_downloader import DEFAULT_TURBO_DIT_MODEL
from acestep.ui.gradio.events.generation.llm_auto_init import ensure_llm_ready
from acestep.ui.gradio.events.generation.quantization import (
    default_quantization_value,
    select_quantization_value,
)


def _resolve_project_root() -> str:
    """Resolve the ACE-Step install root used for local model discovery."""

    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if project_root:
        return str(Path(project_root).expanduser().resolve())
    return str(Path(__file__).resolve().parents[5])


def _last_init_params(handler: Any) -> dict[str, Any]:
    """Return a copy of a handler's last initialization parameters."""

    params = getattr(handler, "last_init_params", None) or {}
    return dict(params) if isinstance(params, dict) else {}


def _default_dit_quantization(device: str) -> str | None:
    """Return the GPU-tier default DiT quantization mode."""

    gpu_config = get_global_gpu_config()
    default_value = default_quantization_value(
        getattr(gpu_config, "quantization_default", None)
    )
    return select_quantization_value(default_value, device=device)


def _dit_init_params(dit_handler: Any, config_path: str | None) -> dict[str, Any]:
    """Build initialization parameters for dataset DiT auto-init."""

    params = _last_init_params(dit_handler)
    gpu_config = get_global_gpu_config()
    selected_model = str(config_path or params.get("config_path") or DEFAULT_TURBO_DIT_MODEL)
    selected_model = selected_model.strip() or DEFAULT_TURBO_DIT_MODEL
    device = str(params.get("device") or "auto").strip() or "auto"
    compile_default = getattr(gpu_config, "compile_model_default", False)
    offload_default = getattr(gpu_config, "offload_to_cpu_default", False)
    offload_dit_default = getattr(gpu_config, "offload_dit_to_cpu_default", False)
    quantization = params.get("quantization")
    if "quantization" not in params:
        quantization = _default_dit_quantization(device)

    return {
        "project_root": str(params.get("project_root") or _resolve_project_root()),
        "config_path": selected_model,
        "device": device,
        "use_flash_attention": bool(params.get("use_flash_attention", False)),
        "compile_model": bool(params.get("compile_model", compile_default)),
        "offload_to_cpu": bool(params.get("offload_to_cpu", offload_default)),
        "offload_dit_to_cpu": bool(
            params.get("offload_dit_to_cpu", offload_dit_default)
        ),
        "quantization": quantization,
        "prefer_source": params.get("prefer_source"),
        "use_mlx_dit": bool(params.get("use_mlx_dit", True)),
        "vae_checkpoint": params.get("vae_checkpoint"),
    }


def _loaded_dit_matches_selection(dit_handler: Any, config_path: str | None) -> bool:
    """Return whether the loaded DiT model matches the requested selection."""

    if getattr(dit_handler, "model", None) is None:
        return False
    requested_model = str(config_path or "").strip()
    if not requested_model:
        return True
    params = _last_init_params(dit_handler)
    return str(params.get("config_path") or "").strip() == requested_model


def ensure_dit_ready(
    dit_handler: Any,
    *,
    config_path: str | None = None,
    training_safe: bool = False,
) -> tuple[bool, str]:
    """Ensure the DiT runtime is initialized for dataset work."""

    if dit_handler is None:
        return False, "DiT handler is unavailable."
    if _loaded_dit_matches_selection(dit_handler, config_path):
        return True, ""

    initialize_service = getattr(dit_handler, "initialize_service", None)
    if not callable(initialize_service):
        return False, "Model not initialized. Please initialize the service first."

    params = _dit_init_params(dit_handler, config_path)
    if training_safe:
        params["quantization"] = None
        params["compile_model"] = False
    logger.info(
        "[training_dataset] Auto-initializing DiT service for dataset action: {}",
        params["config_path"],
    )
    try:
        status, ok = initialize_service(**params)
    except Exception as exc:
        logger.exception("Dataset DiT auto-initialization failed")
        return False, f"Failed to auto-initialize DiT service: {exc!s}"
    if not ok:
        return False, status
    return True, f"DiT service initialized automatically.\n{status}"


def _llm_auto_init_params(llm_handler: Any) -> dict[str, Any]:
    """Build initialization parameters for dataset LM auto-init."""

    params = _last_init_params(llm_handler)
    gpu_config = get_global_gpu_config()
    offload_default = getattr(gpu_config, "offload_to_cpu_default", False)
    return {
        "lm_model_path": params.get("lm_model_path"),
        "backend": params.get("backend"),
        "device": params.get("device") or "auto",
        "offload_to_cpu": bool(params.get("offload_to_cpu", offload_default)),
    }


def ensure_training_services_ready(
    dit_handler: Any,
    llm_handler: Any,
    *,
    require_llm: bool,
    config_path: str | None = None,
) -> tuple[bool, str]:
    """Ensure dataset actions have the required model services."""

    status_lines: list[str] = []
    dit_ok, dit_status = ensure_dit_ready(dit_handler, config_path=config_path)
    if dit_status:
        status_lines.append(dit_status)
    if not dit_ok:
        return False, "\n".join(status_lines)

    if not require_llm:
        return True, "\n".join(status_lines)

    llm_ok, llm_status = ensure_llm_ready(
        llm_handler,
        **_llm_auto_init_params(llm_handler),
    )
    if llm_status:
        status_lines.append(llm_status)
    if not llm_ok:
        return False, "\n".join(status_lines)

    return True, "\n".join(status_lines)
