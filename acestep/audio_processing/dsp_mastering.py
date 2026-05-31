"""Pre-mastering DSP stages for ACE-Step audio processing."""

from __future__ import annotations

import math

import numpy as np
import scipy.signal as signal
from loguru import logger

from .dsp_utils import assign_channel, channel, channel_count, limit_peak
from .dynamics import compress_signal, compression_gain


def multiband_compressor(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Apply gentle three-band compression to low, mid, and high bands."""

    if intensity <= 0 or sample_rate <= 0:
        return audio
    nyquist = sample_rate / 2.0
    low_cut = min(250.0 / nyquist, 0.95)
    high_cut = min(4000.0 / nyquist, 0.95)
    if low_cut >= high_cut:
        return audio
    b_low, a_low = signal.butter(2, low_cut, btype="low")
    b_high, a_high = signal.butter(2, high_cut, btype="low")
    result = audio.copy()
    threshold = -18.0 + (1.0 - float(intensity)) * 10.0
    ratio = 1.5 + float(intensity) * 1.5
    for channel_index in range(channel_count(audio)):
        data = channel(result, channel_index)
        low = signal.lfilter(b_low, a_low, data)
        below_high = signal.lfilter(b_high, a_high, data)
        mid = below_high - low
        high = data - below_high
        out = (
            compress_signal(low, threshold - 2.0, ratio, 30.0, 200.0, sample_rate)
            + compress_signal(mid, threshold, ratio * 0.8, 15.0, 150.0, sample_rate)
            + compress_signal(high, threshold + 2.0, ratio * 0.6, 5.0, 80.0, sample_rate)
        )
        assign_channel(result, channel_index, out)
    return limit_peak(result, 0.99)


def tape_saturation(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Emulate tape-style saturation with a light high-frequency rolloff."""

    if intensity <= 0:
        return audio
    drive = 1.0 + float(intensity) * 3.0
    driven = audio * drive
    saturated = np.where(driven >= 0.0, np.tanh(driven), np.tanh(driven * 0.85)) / drive
    blend = float(intensity) * 0.5
    result = audio * (1.0 - blend) + saturated * blend
    nyquist = max(sample_rate / 2.0, 1.0)
    cutoff = min(15000.0 / nyquist, 0.95)
    if cutoff < 0.95:
        b_filter, a_filter = signal.butter(1, cutoff, btype="low")
        wet = min(float(intensity) * 0.2, 1.0)
        for channel_index in range(channel_count(result)):
            data = channel(result, channel_index)
            filtered = signal.lfilter(b_filter, a_filter, data)
            assign_channel(result, channel_index, data * (1.0 - wet) + filtered * wet)
    return limit_peak(result, 0.99)


def glue_compressor(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Apply linked stereo bus compression for mix cohesion."""

    if intensity <= 0 or sample_rate <= 0:
        return audio
    threshold = -18.0 + (1.0 - float(intensity)) * 10.0
    ratio = 1.5 + float(intensity) * 1.5
    attack = 30.0 - float(intensity) * 20.0
    release = 250.0 - float(intensity) * 100.0
    if audio.ndim < 2:
        return limit_peak(
            compress_signal(audio, threshold, ratio, attack, release, sample_rate),
            0.99,
        )

    envelope_source = np.max(np.abs(audio), axis=1)
    gain = compression_gain(envelope_source, threshold, ratio, attack, release, sample_rate)
    return limit_peak(audio * gain[:, None], 0.99)


def midside_eq(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Shape center and side tone with subtle mid/side EQ."""

    if intensity <= 0 or audio.ndim < 2 or audio.shape[1] < 2 or sample_rate <= 0:
        return audio
    left = audio[:, 0]
    right = audio[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    nyquist = sample_rate / 2.0
    mid = _blend_filter(mid, 40.0 / nyquist, "high", intensity * 0.15)
    if 5000.0 < nyquist:
        b_pres, a_pres = signal.butter(
            2,
            [min(3000.0 / nyquist, 0.9), min(5000.0 / nyquist, 0.95)],
            btype="band",
        )
        mid = mid + signal.lfilter(b_pres, a_pres, mid) * intensity * 0.08
    if 12000.0 < nyquist:
        side = side + _filtered(side, 12000.0 / nyquist, "high") * intensity * 0.15
    side = _blend_filter(side, 120.0 / nyquist, "high", intensity * 0.25)
    result = audio.copy()
    result[:, 0] = mid + side
    result[:, 1] = mid - side
    return limit_peak(result, 0.99)


def soft_clipper(audio: np.ndarray, intensity: float) -> np.ndarray:
    """Drive a transparent soft clipper for controlled loudness."""

    if intensity <= 0:
        return audio
    drive = 1.0 + float(intensity) * 1.5
    driven = audio * drive
    threshold = 0.85
    clipped = np.where(
        np.abs(driven) <= threshold,
        driven,
        np.sign(driven)
        * (threshold + (1.0 - threshold) * np.tanh((np.abs(driven) - threshold) / 0.15)),
    )
    clipped = clipped / drive
    blend = float(intensity) * 0.6
    return limit_peak(audio * (1.0 - blend) + clipped * blend, 0.99)


def lufs_normalize(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """Normalize integrated loudness and keep peak below -1 dBFS."""

    current = measure_lufs(audio, sample_rate)
    if not math.isfinite(current):
        return audio
    gain = 10.0 ** ((float(target_lufs) - current) / 20.0)
    result = audio * gain
    limit = 10.0 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > limit:
        result = result / peak * limit
    return np.clip(result, -1.0, 1.0)


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure integrated LUFS, falling back to RMS loudness if needed."""

    if not audio.size:
        return float("-inf")
    try:
        import pyloudnorm as pyln

        return float(pyln.Meter(int(sample_rate)).integrated_loudness(audio))
    except Exception as exc:
        logger.debug("LUFS meter unavailable, using RMS fallback: {}", exc)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        return 20.0 * math.log10(max(rms, 1e-12))


def _blend_filter(data: np.ndarray, cutoff: float, kind: str, wet: float) -> np.ndarray:
    """Blend a filtered signal with the original."""

    filtered = _filtered(data, cutoff, kind)
    return data * (1.0 - wet) + filtered * wet


def _filtered(data: np.ndarray, cutoff: float, kind: str) -> np.ndarray:
    """Filter one signal with safe normalized cutoff handling."""

    b_filter, a_filter = signal.butter(1, min(max(cutoff, 1e-5), 0.95), btype=kind)
    return signal.lfilter(b_filter, a_filter, data)

