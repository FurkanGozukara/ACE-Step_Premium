"""On-demand 5Hz LM initialization helpers for LM-assisted UI actions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from acestep.gpu_config import (
    find_best_lm_model_on_disk,
    get_global_gpu_config,
    resolve_lm_backend,
)
from acestep.model_downloader import (
    DEFAULT_LM_MODEL,
    ensure_lm_model,
    get_models_dir,
)
from acestep.torch_compile_workers import normalize_compile_threads
from acestep.ui.gradio.i18n import t


def _resolve_project_root() -> Path:
    """Resolve the ACE-Step install root for local model access."""

    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if project_root:
        return Path(project_root).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


def _resolve_requested_lm_model(llm_handler: Any, lm_model_path: str | None) -> str:
    """Return the LM model name to use for an on-demand action."""

    requested = str(lm_model_path or "").strip()
    if requested:
        return requested

    try:
        gpu_config = get_global_gpu_config()
        recommended_lm = getattr(gpu_config, "recommended_lm_model", "")
        disk_models = list(llm_handler.get_available_5hz_lm_models())
        selected_model = find_best_lm_model_on_disk(recommended_lm, disk_models)
        if selected_model or recommended_lm:
            return selected_model or recommended_lm
    except Exception:
        pass

    try:
        default_model = str(llm_handler.get_default_lm_model() or "").strip()
        if default_model:
            return default_model
    except Exception:
        pass

    return DEFAULT_LM_MODEL


def _llm_needs_reinit(
    llm_handler: Any,
    *,
    lm_model_path: str,
    backend: str,
    device: str,
    offload_to_cpu: bool,
    compile_model: bool,
    compile_threads: int = 8,
) -> bool:
    """Return whether the current LM runtime must be initialized or refreshed."""

    if llm_handler is None:
        return True
    if not getattr(llm_handler, "llm_initialized", False):
        return True

    last_init_params = getattr(llm_handler, "last_init_params", None) or {}
    if not last_init_params:
        if getattr(llm_handler, "llm_initialized", False) and not hasattr(
            llm_handler, "initialize"
        ):
            return False
        return True
    if last_init_params.get("lm_model_path") != lm_model_path:
        return True
    if last_init_params.get("backend") != backend:
        return True
    if device != "auto" and last_init_params.get("device") != device:
        return True
    if bool(last_init_params.get("offload_to_cpu")) != bool(offload_to_cpu):
        return True
    if bool(last_init_params.get("compile_model")) != bool(compile_model):
        return True
    if bool(compile_model) and normalize_compile_threads(
        last_init_params.get("compile_threads")
    ) != normalize_compile_threads(compile_threads):
        return True
    return False


def ensure_llm_ready(
    llm_handler: Any,
    *,
    lm_model_path: str | None,
    backend: str | None,
    device: str | None,
    offload_to_cpu: bool,
) -> tuple[bool, str]:
    """Ensure the 5Hz LM is available for a UI action.

    Returns ``(ok, status_message)``. The status message is empty when no
    initialization work was needed.
    """

    if llm_handler is None:
        return False, "5Hz LM handler is unavailable."
    if (
        not getattr(llm_handler, "llm_initialized", False)
        and not hasattr(llm_handler, "initialize")
    ):
        return False, t("messages.lm_not_initialized")

    gpu_config = get_global_gpu_config()
    requested_model = _resolve_requested_lm_model(llm_handler, lm_model_path)
    requested_backend = str(
        backend or getattr(gpu_config, "recommended_backend", "pt") or "pt"
    ).strip().lower() or "pt"
    resolved_backend = resolve_lm_backend(requested_backend, gpu_config)
    resolved_device = str(device or "auto").strip() or "auto"
    resolved_offload = bool(offload_to_cpu)
    resolved_compile = os.getenv("ACESTEP_COMPILE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    resolved_compile_threads = normalize_compile_threads(None)

    if not _llm_needs_reinit(
        llm_handler,
        lm_model_path=requested_model,
        backend=resolved_backend,
        device=resolved_device,
        offload_to_cpu=resolved_offload,
        compile_model=resolved_compile,
        compile_threads=resolved_compile_threads,
    ):
        return True, ""

    project_root = _resolve_project_root()
    models_dir = get_models_dir(project_root=project_root)

    download_ok, download_status = ensure_lm_model(
        model_name=requested_model,
        checkpoints_dir=models_dir,
    )
    if not download_ok:
        return (
            False,
            f"Failed to prepare 5Hz LM model '{requested_model}'.\n{download_status}",
        )

    init_status, init_ok = llm_handler.initialize(
        checkpoint_dir=str(models_dir),
        lm_model_path=requested_model,
        backend=resolved_backend,
        device=resolved_device,
        offload_to_cpu=resolved_offload,
        dtype=None,
        compile_model=resolved_compile,
        compile_threads=resolved_compile_threads,
    )
    llm_handler.last_init_params = {
        "lm_model_path": requested_model,
        "backend": resolved_backend,
        "device": resolved_device,
        "offload_to_cpu": resolved_offload,
        "compile_model": resolved_compile,
        "compile_threads": resolved_compile_threads,
    }
    if not init_ok:
        return False, init_status

    return True, f"5Hz LM initialized automatically.\n{init_status}"
