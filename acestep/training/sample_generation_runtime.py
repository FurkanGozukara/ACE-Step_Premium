"""Runtime helpers for in-process LoRA sample generation."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn


@contextmanager
def sample_runtime_context(handler: Any, *, enabled: bool) -> Iterator[None]:
    """Temporarily switch the handler to low-spike sample generation mode."""

    previous_offload = getattr(handler, "offload_to_cpu", False)
    previous_dit_offload = getattr(handler, "offload_dit_to_cpu", False)
    decoder = getattr(getattr(handler, "model", None), "decoder", None)
    decoder_was_training = bool(getattr(decoder, "training", False))
    try:
        if enabled:
            handler.offload_to_cpu = True
            handler.offload_dit_to_cpu = False
        if isinstance(decoder, nn.Module):
            decoder.eval()
        yield
    finally:
        if isinstance(decoder, nn.Module):
            decoder.train(decoder_was_training)
        handler.offload_to_cpu = previous_offload
        handler.offload_dit_to_cpu = previous_dit_offload
        release_memory()


def serializable_audios(audios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-safe audio metadata."""

    return [
        {
            "path": str(audio.get("path") or ""),
            "key": str(audio.get("key") or ""),
            "sample_rate": int(audio.get("sample_rate") or 0),
        }
        for audio in audios
    ]


def cuda_peak_gb() -> float:
    """Return the current process CUDA peak allocation in GiB."""

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def release_memory() -> None:
    """Release Python and CUDA caches after sample generation."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
