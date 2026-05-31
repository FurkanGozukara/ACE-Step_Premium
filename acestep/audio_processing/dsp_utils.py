"""Shared low-level helpers for audio-processing DSP modules."""

from __future__ import annotations

import numpy as np


def limit_peak(audio: np.ndarray, peak_target: float) -> np.ndarray:
    """Scale audio down only when it exceeds the target peak."""

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio / peak * peak_target if peak > peak_target else audio


def channel_count(audio: np.ndarray) -> int:
    """Return the number of channels in a mono or channel-last array."""

    return audio.shape[1] if audio.ndim > 1 else 1


def channel(audio: np.ndarray, index: int) -> np.ndarray:
    """Return a mutable channel view from a mono or channel-last array."""

    return audio[:, index] if audio.ndim > 1 else audio


def assign_channel(audio: np.ndarray, index: int, values: np.ndarray) -> None:
    """Assign channel data into a mono or channel-last array."""

    if audio.ndim > 1:
        audio[:, index] = values
    else:
        audio[:] = values
