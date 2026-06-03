"""Spectral residual construction for ACE-Step Extract."""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_N_FFT = 4096
MIN_SPECTRAL_SAMPLES = 512
SPECTRAL_FLOOR_RATIO = 0.04
SPECTRAL_STRENGTH = 1.35


def build_remaining_audio(
    source: torch.Tensor,
    extracted: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build remaining audio from a source waveform and extracted stem.

    ACE-Step Extract output is not guaranteed to be phase-coherent with the
    input mixture.  Direct waveform subtraction can therefore sound nearly
    unchanged.  This helper subtracts extracted-stem magnitude from the source
    spectrum while preserving the source phase, which is closer to separation
    residual behavior.

    Args:
        source: Source mixture tensor in channel-first layout.
        extracted: Extracted stem tensor in channel-first layout.

    Returns:
        Tuple of remaining audio and JSON-safe subtraction metadata.
    """

    samples = int(source.shape[-1])
    if samples < MIN_SPECTRAL_SAMPLES:
        remaining = torch.clamp(source - extracted, min=-1.0, max=1.0)
        return remaining, {"method": "waveform_subtraction_short_audio"}

    n_fft = _analysis_size(samples)
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, dtype=source.dtype, device=source.device)
    channels = []
    gains: list[float] = []
    for channel_idx in range(int(source.shape[0])):
        remaining_channel, gain = _subtract_channel_spectrum(
            source[channel_idx],
            extracted[channel_idx],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        )
        channels.append(remaining_channel)
        gains.append(float(gain))

    metadata = {
        "method": "source_phase_spectral_subtraction",
        "n_fft": n_fft,
        "hop_length": hop_length,
        "strength": SPECTRAL_STRENGTH,
        "floor_ratio": SPECTRAL_FLOOR_RATIO,
        "channel_gains": gains,
    }
    return torch.clamp(torch.stack(channels), min=-1.0, max=1.0), metadata


def _subtract_channel_spectrum(
    source: torch.Tensor,
    extracted: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Return one source-phase residual channel and its matched gain."""

    source_spec = torch.stft(
        source,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    extracted_spec = torch.stft(
        extracted,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    source_mag = source_spec.abs()
    extracted_mag = extracted_spec.abs()
    gain = _matched_spectral_gain(source_mag, extracted_mag)
    remaining_mag = torch.clamp(
        source_mag - (SPECTRAL_STRENGTH * gain * extracted_mag),
        min=SPECTRAL_FLOOR_RATIO * source_mag,
    )
    residual_spec = source_spec * (remaining_mag / source_mag.clamp_min(1e-8))
    remaining = torch.istft(
        residual_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=int(source.shape[-1]),
    )
    return remaining, float(gain)


def _matched_spectral_gain(source_mag: torch.Tensor, extracted_mag: torch.Tensor) -> torch.Tensor:
    """Return a robust gain that maps extracted magnitude onto the source."""

    threshold = _sampled_quantile(extracted_mag, 0.80)
    mask = extracted_mag > threshold
    if not torch.any(mask):
        return torch.tensor(1.0, dtype=source_mag.dtype, device=source_mag.device)
    numerator = torch.sum(source_mag[mask] * extracted_mag[mask])
    denominator = torch.sum(extracted_mag[mask] * extracted_mag[mask]).clamp_min(1e-8)
    return torch.clamp(numerator / denominator, min=0.25, max=4.0)


def _sampled_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """Return an approximate quantile without materializing huge sort work."""

    flattened = values.flatten()
    step = max(1, int(flattened.numel()) // 500_000)
    return torch.quantile(flattened[::step], q)


def _analysis_size(samples: int) -> int:
    """Return a safe STFT size for the given sample count."""

    size = 1
    limit = min(DEFAULT_N_FFT, samples)
    while size * 2 <= limit:
        size *= 2
    return max(MIN_SPECTRAL_SAMPLES, size)
