"""Mel-feature extraction helpers for DiffPitcher."""

from __future__ import annotations

import librosa
import numpy as np
import pyworld as pw

from .diffpitcher_audio_io import pad_count, pad_for_stft, trim_to_hop
from .diffpitcher_constants import (
    DIFFPITCHER_HOP,
    DIFFPITCHER_MEL_BASIS,
    DIFFPITCHER_N_FFT,
    DIFFPITCHER_SAMPLE_RATE,
)


def world_mel_from_wav(wav: np.ndarray) -> np.ndarray:
    """Return DiffPitcher WORLD-depitched log mel features."""

    wav = trim_to_hop(wav)
    wav64 = world_float64(wav)
    f0, time_axis = pw.dio(wav64, DIFFPITCHER_SAMPLE_RATE)
    f0 = pw.stonemask(wav64, f0, time_axis, DIFFPITCHER_SAMPLE_RATE)
    spectral = pw.cheaptrick(wav64, f0, time_axis, DIFFPITCHER_SAMPLE_RATE)
    aperiodicity = pw.d4c(wav64, f0, time_axis, DIFFPITCHER_SAMPLE_RATE)
    depitched = pw.synthesize(
        f0 * 0.0,
        spectral,
        aperiodicity,
        DIFFPITCHER_SAMPLE_RATE,
    )[: len(wav)]
    return log_mel_from_wav(np.asarray(depitched, dtype=np.float32))


def log_mel_from_wav(wav: np.ndarray) -> np.ndarray:
    """Return DiffPitcher log mel features."""

    wav = trim_to_hop(wav)
    stft = librosa.stft(
        pad_for_stft(wav),
        n_fft=DIFFPITCHER_N_FFT,
        hop_length=DIFFPITCHER_HOP,
        win_length=DIFFPITCHER_N_FFT,
        window="hann",
        center=False,
    )
    magnitude = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
    mel = np.matmul(DIFFPITCHER_MEL_BASIS, magnitude)
    mel = pad_mel_frames(mel, pad_mode="minimum")
    return np.log(np.clip(mel, a_min=1e-5, a_max=None)).astype(np.float32)


def pad_mel_frames(mel: np.ndarray, *, pad_mode: str) -> np.ndarray:
    """Pad mel frames to a multiple of eight."""

    pad = pad_count(mel.shape[-1]) - mel.shape[-1]
    if pad <= 0:
        return mel
    return np.pad(mel, ((0, 0), (0, pad)), mode=pad_mode)


def world_float64(wav: np.ndarray) -> np.ndarray:
    """Return WORLD-compatible float64 audio."""

    quantized = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    return (quantized.astype(np.float64) / 32767.0).astype(np.float64)
