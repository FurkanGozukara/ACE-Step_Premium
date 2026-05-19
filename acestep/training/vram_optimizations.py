"""VRAM optimization helpers for LoRA training."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn
from loguru import logger


_NON_DECODER_MODULES = (
    "music_encoder",
    "lyric_encoder",
    "timbre_encoder",
    "condition_projection",
    "vae",
    "text_encoder",
    "attention_pooler",
)


def offload_non_decoder_modules(model: nn.Module) -> tuple[str, ...]:
    """Move modules unused by preprocessed-tensor LoRA training to CPU."""

    moved: list[str] = []
    for name in _NON_DECODER_MODULES:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            module.to("cpu")
            moved.append(name)
    _release_cuda_cache()
    return tuple(moved)


def offload_handler_training_modules(handler: Any) -> tuple[str, ...]:
    """Move handler-owned inference modules unused by tensor LoRA training to CPU."""

    moved: list[str] = []
    for name in ("vae", "text_encoder"):
        module = getattr(handler, name, None)
        if isinstance(module, nn.Module):
            module.to("cpu")
            moved.append(name)
    _release_cuda_cache()
    return tuple(moved)


def cast_training_parameter_dtypes(
    module: nn.Module,
    *,
    frozen_dtype: torch.dtype,
    keep_frozen_in_compute_dtype: bool,
) -> tuple[int, int]:
    """Cast frozen base weights cheaply while keeping trainable adapters in fp32.

    Args:
        module: Decoder or wrapped decoder to update in-place.
        frozen_dtype: Target dtype for frozen floating-point weights.
        keep_frozen_in_compute_dtype: Whether frozen weights should be cast.

    Returns:
        Tuple of ``(trainable_casted, frozen_casted)``.
    """

    trainable_casted = 0
    frozen_casted = 0
    for param in module.parameters():
        if not param.is_floating_point() or _is_float8(param.dtype):
            continue
        if param.requires_grad:
            if param.dtype != torch.float32:
                with torch.no_grad():
                    param.data = param.data.float()
                trainable_casted += 1
            continue
        if keep_frozen_in_compute_dtype and param.dtype != frozen_dtype:
            with torch.no_grad():
                param.data = param.data.to(dtype=frozen_dtype)
            frozen_casted += 1
    return trainable_casted, frozen_casted


def apply_training_fp8_scaled(
    model: nn.Module,
    *,
    checkpoint_path: str | None,
    device: str | torch.device,
) -> str:
    """Apply scaled FP8 to frozen decoder linears for LoRA training."""

    from acestep.core.generation.handler.fp8_scaled_quantization import (
        apply_fp8_scaled_quantization,
    )

    apply_fp8_scaled_quantization(
        model,
        checkpoint_path=checkpoint_path,
        device=device,
        skip_trainable=True,
    )
    _release_cuda_cache()
    return "Applied scaled FP8 to frozen decoder Linear weights"


@contextmanager
def sample_generation_vram_guard(
    module: Any,
    *,
    enabled: bool,
    target_device: torch.device | str,
) -> Iterator[None]:
    """Temporarily move the training decoder to CPU while samples generate."""

    decoder = getattr(getattr(module, "model", None), "decoder", None)
    if not enabled or not isinstance(decoder, nn.Module):
        yield
        return

    logger.info("Moving training decoder to CPU before checkpoint sample generation")
    decoder.to("cpu")
    _release_cuda_cache()
    try:
        yield
    finally:
        logger.info("Moving training decoder back to {} after sample generation", target_device)
        decoder.to(target_device)
        _release_cuda_cache()
        try:
            decoder.train()
        except Exception:
            pass


def cuda_peak_gb() -> float:
    """Return current-process CUDA peak allocation in GiB."""

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_cuda_peak() -> None:
    """Reset current-process CUDA peak statistics when CUDA is available."""

    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _release_cuda_cache() -> None:
    """Release Python and CUDA caches after large VRAM transitions."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _is_float8(dtype: torch.dtype) -> bool:
    """Return whether *dtype* is one of PyTorch's FP8 dtypes."""

    return str(dtype).startswith("torch.float8")
