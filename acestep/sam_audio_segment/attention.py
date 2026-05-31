"""SAM-Audio scaled-dot-product attention backend selection."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn.functional as F

SAM_ATTENTION_AUTO = "auto"
SAM_ATTENTION_FLASH = "flash"
SAM_ATTENTION_CUDNN = "cudnn"
SAM_ATTENTION_EFFICIENT = "efficient"
SAM_ATTENTION_MATH = "math"

SAM_ATTENTION_CHOICES: tuple[tuple[str, str], ...] = (
    ("Auto SDPA", SAM_ATTENTION_AUTO),
    ("Memory Efficient SDPA", SAM_ATTENTION_EFFICIENT),
    ("Flash Preferred SDPA", SAM_ATTENTION_FLASH),
    ("cuDNN SDPA", SAM_ATTENTION_CUDNN),
    ("Math SDPA", SAM_ATTENTION_MATH),
)

_ALLOWED_BACKENDS = {
    SAM_ATTENTION_AUTO,
    SAM_ATTENTION_FLASH,
    SAM_ATTENTION_CUDNN,
    SAM_ATTENTION_EFFICIENT,
    SAM_ATTENTION_MATH,
}

_ACTIVE_BACKEND: ContextVar[str] = ContextVar("sam_attention_backend", default=SAM_ATTENTION_AUTO)
_ACTIVE_STATS: ContextVar[dict[str, int] | None] = ContextVar(
    "sam_attention_stats",
    default=None,
)


def normalize_attention_backend(value: Any) -> str:
    """Return a supported SAM-Audio attention backend name."""

    normalized = str(value or "").strip().lower()
    return normalized if normalized in _ALLOWED_BACKENDS else SAM_ATTENTION_AUTO


@contextmanager
def attention_backend_context(backend: str):
    """Set the active SAM-Audio attention backend for vendored attention calls."""

    token = _ACTIVE_BACKEND.set(normalize_attention_backend(backend))
    try:
        yield
    finally:
        _ACTIVE_BACKEND.reset(token)


def reset_attention_stats(backend: str) -> None:
    """Reset SDPA backend counters for the current SAM-Audio request."""

    _ACTIVE_STATS.set(
        {
            "backend": normalize_attention_backend(backend),
            "calls": 0,
            "auto_calls": 0,
            "flash_calls": 0,
            "cudnn_calls": 0,
            "efficient_calls": 0,
            "math_calls": 0,
            "fallback_calls": 0,
            "flash_unavailable": 0,
            "cudnn_unavailable": 0,
            "efficient_unavailable": 0,
            "contiguous_conversions": 0,
            "trivial_masks_removed": 0,
        }
    )


def attention_stats() -> dict[str, int]:
    """Return SDPA backend counters for the current SAM-Audio request."""

    stats = _ACTIVE_STATS.get()
    return dict(stats) if stats is not None else {}


def sam_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run SDPA with the selected SAM-Audio backend and safe fallbacks."""

    _count("calls")
    backend = _ACTIVE_BACKEND.get()
    if query.device.type != "cuda" or backend == SAM_ATTENTION_AUTO:
        _count("auto_calls")
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    if backend == SAM_ATTENTION_FLASH:
        effective_mask = _drop_trivial_mask(attn_mask)
        return _run_flash_preferred(query, key, value, effective_mask)
    if backend == SAM_ATTENTION_CUDNN:
        effective_mask = _drop_trivial_mask(attn_mask)
        return _run_cudnn_or_fallback(query, key, value, effective_mask)
    if backend == SAM_ATTENTION_EFFICIENT:
        effective_mask = _drop_trivial_mask(attn_mask)
        return _run_efficient_or_fallback(query, key, value, effective_mask)
    return _run_with_backend(query, key, value, attn_mask, SAM_ATTENTION_MATH)


def sdpa_backend_context(backend: str, device: torch.device):
    """Return a strict PyTorch SDPA backend context for the requested mode."""

    if device.type != "cuda":
        return nullcontext()
    normalized = normalize_attention_backend(backend)
    if normalized == SAM_ATTENTION_AUTO:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if normalized == SAM_ATTENTION_FLASH:
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    if normalized == SAM_ATTENTION_CUDNN:
        return sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
    if normalized == SAM_ATTENTION_EFFICIENT:
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    return sdpa_kernel(SDPBackend.MATH)


def _run_flash_preferred(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run Flash SDPA when supported, otherwise cascade to fast safe backends."""

    if not _is_unavailable(SAM_ATTENTION_FLASH):
        try:
            return _run_with_backend(query, key, value, attn_mask, SAM_ATTENTION_FLASH)
        except RuntimeError:
            _mark_unavailable(SAM_ATTENTION_FLASH)
            _count("fallback_calls")
    return _run_cudnn_or_fallback(query, key, value, attn_mask)


def _run_cudnn_or_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run strict cuDNN SDPA when supported, otherwise fall back."""

    if not _is_unavailable(SAM_ATTENTION_CUDNN):
        try:
            return _run_with_backend(query, key, value, attn_mask, SAM_ATTENTION_CUDNN)
        except RuntimeError:
            _mark_unavailable(SAM_ATTENTION_CUDNN)
            _count("fallback_calls")
    return _run_efficient_or_fallback(query, key, value, attn_mask)


def _run_efficient_or_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run strict memory-efficient SDPA when supported, otherwise fall back."""

    if not _is_unavailable(SAM_ATTENTION_EFFICIENT):
        try:
            return _run_with_backend(query, key, value, attn_mask, SAM_ATTENTION_EFFICIENT)
        except RuntimeError:
            _mark_unavailable(SAM_ATTENTION_EFFICIENT)
            _count("fallback_calls")
    _count("auto_calls")
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)


def _run_with_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    backend: str,
) -> torch.Tensor:
    """Run SDPA under one strict PyTorch backend."""

    if backend != SAM_ATTENTION_MATH:
        query, key, value = _contiguous_qkv(query, key, value)
    with sdpa_backend_context(backend, query.device):
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    _count(f"{backend}_calls")
    return result


def _drop_trivial_mask(attn_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Drop all-true boolean masks so Flash SDPA can run on unmasked attention."""

    if attn_mask is None or attn_mask.dtype != torch.bool:
        return attn_mask
    if bool(attn_mask.all().item()):
        _count("trivial_masks_removed")
        return None
    return attn_mask


def _contiguous_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contiguous Q/K/V tensors for CUDA optimized SDPA kernels."""

    if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
        _count("contiguous_conversions")
    return query.contiguous(), key.contiguous(), value.contiguous()


def _count(name: str) -> None:
    """Increment an attention counter when stats are active."""

    stats = _ACTIVE_STATS.get()
    if stats is not None:
        stats[name] = int(stats.get(name, 0)) + 1


def _mark_unavailable(backend: str) -> None:
    """Mark one strict SDPA backend unavailable for this request."""

    stats = _ACTIVE_STATS.get()
    if stats is not None:
        stats[f"{backend}_unavailable"] = 1


def _is_unavailable(backend: str) -> bool:
    """Return whether one strict SDPA backend already failed this request."""

    stats = _ACTIVE_STATS.get()
    return bool(stats and stats.get(f"{backend}_unavailable"))
