"""Memory telemetry helpers for SAM-Audio runs."""

from __future__ import annotations

import torch


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak memory counters when available."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def peak_memory_metrics(device: torch.device) -> dict[str, float]:
    """Return peak CUDA memory metrics in GiB."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    except Exception:
        return {}
    return {
        "peak_allocated_gb": round(float(allocated), 3),
        "peak_reserved_gb": round(float(reserved), 3),
    }
