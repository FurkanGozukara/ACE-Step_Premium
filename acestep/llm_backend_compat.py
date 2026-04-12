"""Compatibility helpers for optional 5Hz LM backends."""

import importlib
import sys


def _has_working_triton_installation() -> bool:
    """Return whether the Triton modules required by nano-vllm import cleanly."""
    try:
        importlib.import_module("triton")
        importlib.import_module("triton.language")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _has_torch_nccl_support() -> bool:
    """Return whether the installed PyTorch distributed build supports NCCL."""

    try:
        dist = importlib.import_module("torch.distributed")
    except (ImportError, ModuleNotFoundError):
        return False

    is_available = getattr(dist, "is_available", None)
    if callable(is_available) and not is_available():
        return False

    is_nccl_available = getattr(dist, "is_nccl_available", None)
    if callable(is_nccl_available):
        try:
            return bool(is_nccl_available())
        except Exception:
            return False
    return False


def get_vllm_preflight_warning(*, device: str, platform: str | None = None) -> str | None:
    """Return a user-facing warning when vLLM should be skipped before initialization.

    Args:
        device: The resolved device string for LM initialization.
        platform: Optional platform override for tests. Defaults to ``sys.platform``.

    Returns:
        A warning string when vLLM should fall back to PyTorch, otherwise ``None``.
    """
    active_platform = sys.platform if platform is None else platform
    if device != "cuda":
        return None
    if not _has_torch_nccl_support():
        if active_platform == "win32":
            return (
                "vLLM backend is unavailable on Windows because the installed "
                "PyTorch distributed build does not include NCCL support. "
                "Falling back to the PyTorch backend. Use --backend pt to suppress "
                "this warning."
            )
        return (
            "vLLM backend is unavailable because the installed PyTorch distributed "
            "build does not include NCCL support. Falling back to the PyTorch "
            "backend. Use --backend pt to suppress this warning."
        )
    if active_platform == "win32" and not _has_working_triton_installation():
        return (
            "vLLM backend is unavailable on Windows because Triton is not installed "
            "or is incompatible. Falling back to the PyTorch backend. "
            "Use --backend pt to suppress this warning."
        )
    return None
