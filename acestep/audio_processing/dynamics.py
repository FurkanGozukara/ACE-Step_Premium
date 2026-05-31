"""Reusable dynamics helpers for audio-processing mastering stages."""

from __future__ import annotations

import numpy as np


def compress_signal(
    data: np.ndarray,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    sample_rate: int,
) -> np.ndarray:
    """Compress one mono signal using envelope gain reduction."""

    return data * compression_gain(
        data,
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        sample_rate,
    )


def compression_gain(
    data: np.ndarray,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    sample_rate: int,
) -> np.ndarray:
    """Return compressor gain for a mono signal or envelope."""

    threshold = 10.0 ** (threshold_db / 20.0)
    attack = np.exp(-1.0 / max(sample_rate * attack_ms / 1000.0, 1.0))
    release = np.exp(-1.0 / max(sample_rate * release_ms / 1000.0, 1.0))
    envelope = np.zeros_like(data, dtype=np.float32)
    env = 0.0
    for index, sample in enumerate(np.abs(data)):
        coeff = attack if sample > env else release
        env = coeff * env + (1.0 - coeff) * float(sample)
        envelope[index] = env
    gain = np.ones_like(envelope, dtype=np.float32)
    mask = envelope > threshold
    if np.any(mask):
        over_db = 20.0 * np.log10(np.clip(envelope[mask] / threshold, 1e-10, None))
        gain[mask] = 10.0 ** (-(over_db * (1.0 - 1.0 / ratio)) / 20.0)
    return gain
