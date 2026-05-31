"""Audio-enhancement DSP stages for generated music post-processing."""

from __future__ import annotations

import numpy as np
import scipy.signal as signal

from .dsp_utils import assign_channel, channel, channel_count, limit_peak


def stereo_depth(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Add subtle left/right phase variation while keeping bass stable."""

    if intensity <= 0 or audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    right = audio[:, 1]
    freq_norm = min(0.02 + float(intensity) * 0.03, 0.09)
    b_filter, a_filter = signal.butter(2, freq_norm, btype="low")
    lowpassed = signal.lfilter(b_filter, a_filter, right)
    phase_shifted = 2.0 * lowpassed - right
    result = audio.copy()
    blend = float(intensity) * 0.25
    result[:, 1] = right * (1.0 - blend) + phase_shifted * blend
    return result


def stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    """Widen stereo image using mid/side processing."""

    if audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    left = audio[:, 0]
    right = audio[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * float(width)
    result = audio.copy()
    result[:, 0] = mid + side
    result[:, 1] = mid - side
    return limit_peak(result, 0.99)


def hf_refinement(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Smooth harsh high-frequency content above 12 kHz."""

    if intensity <= 0 or sample_rate <= 0:
        return audio
    nyquist = sample_rate / 2.0
    if 12000.0 >= nyquist:
        return audio
    b_low, a_low = signal.butter(2, 12000.0 / nyquist, btype="low")
    result = audio.copy()
    for channel_index in range(channel_count(audio)):
        data = channel(audio, channel_index)
        low = signal.lfilter(b_low, a_low, data)
        high = data - low
        kernel_size = max(3, int(sample_rate * 0.001))
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smoothed = np.convolve(high, kernel, mode="same")
        out = low + high * (1.0 - intensity * 0.5) + smoothed * (intensity * 0.5)
        assign_channel(result, channel_index, out)
    return result


def harmonic_enrichment(audio: np.ndarray, intensity: float) -> np.ndarray:
    """Add gentle asymmetric saturation for analog-style warmth."""

    if intensity <= 0:
        return audio
    drive = 1.0 + float(intensity) * 1.5
    driven = audio * drive
    positive = driven - (driven**3) / 3.0
    negative = np.tanh(driven * 0.8)
    saturated = np.where(driven >= 0.0, positive, negative)
    saturated = np.clip(saturated, -1.0, 1.0) / drive
    blend = float(intensity) * 0.2
    return limit_peak(audio * (1.0 - blend) + saturated * blend, 0.99)


def timing_humanizer(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Shift tiny transient regions by sub-millisecond offsets."""

    if intensity <= 0 or sample_rate <= 0 or len(audio) < sample_rate * 0.05:
        return audio
    result = audio.copy()
    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio.copy()
    frame_size = max(16, int(sample_rate * 0.01))
    hop = max(1, frame_size // 2)
    energy = np.array(
        [np.sum(mono[i : i + frame_size] ** 2) for i in range(0, len(mono) - frame_size, hop)]
    )
    if len(energy) < 3:
        return audio
    threshold = float(np.mean(energy) + np.std(energy) * 1.5)
    peaks, _ = signal.find_peaks(
        energy,
        height=threshold,
        distance=max(1, int(0.05 * sample_rate / hop)),
    )
    max_shift = int(sample_rate * 0.0005 * float(intensity))
    if max_shift < 1:
        return audio
    rng = np.random.default_rng(42)
    for peak_idx in peaks:
        _shift_transient_region(result, int(peak_idx * hop), frame_size, max_shift, rng)
    return result


def ambience_shaping(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Add gated low-level pink ambience for a less sterile noise floor."""

    if intensity <= 0 or sample_rate <= 0:
        return audio
    result = audio.copy()
    n_samples = len(audio)
    envelope = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)
    kernel_len = max(1, int(sample_rate * 0.05))
    kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
    gate = np.convolve(envelope, kernel, mode="same")
    gate = np.clip(gate / (np.max(gate) + 1e-10), 0.0, 1.0)
    rng = np.random.default_rng(123)
    noise_level = 10.0 ** ((-96.0 + float(intensity) * 18.0) / 20.0)
    for channel_index in range(channel_count(audio)):
        white = rng.standard_normal(n_samples).astype(np.float32)
        pink = signal.lfilter(
            [0.049922035, -0.095993537, 0.050612699, -0.004709510],
            [1.0, -2.494956002, 2.017265875, -0.522189400],
            white,
        )
        peak = np.max(np.abs(pink))
        if peak > 0:
            pink = pink / peak
        assign_channel(
            result,
            channel_index,
            channel(result, channel_index) + pink * noise_level * gate,
        )
    return result


def _shift_transient_region(
    audio: np.ndarray,
    center: int,
    frame_size: int,
    max_shift: int,
    rng: np.random.Generator,
) -> None:
    """Shift a short region around one transient in-place."""

    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return
    start = max(0, center - frame_size)
    end = min(len(audio), center + frame_size)
    fade_len = max(16, frame_size // 4)
    if end - start < abs(shift) * 2 + fade_len * 2:
        return
    x_new = np.arange(end - start)
    x_old = np.clip(x_new - shift, 0, end - start - 1)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    for channel_index in range(channel_count(audio)):
        region = channel(audio[start:end], channel_index).copy()
        shifted = np.interp(x_new, x_old, region)
        shifted[:fade_len] = region[:fade_len] * (1.0 - fade_in) + shifted[:fade_len] * fade_in
        shifted[-fade_len:] = (
            region[-fade_len:] * (1.0 - fade_out) + shifted[-fade_len:] * fade_out
        )
        assign_channel(audio[start:end], channel_index, shifted)
