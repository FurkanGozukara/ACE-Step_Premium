"""Waveform-level source plus generated-layer mixing for Lego tasks."""

from __future__ import annotations

from typing import List

import torch

from acestep.core.generation.handler.repaint_waveform_splice import (
    _build_waveform_crossfade_mask,
)


def apply_lego_layer_mix(
    pred_wavs: torch.Tensor,
    src_wavs: torch.Tensor,
    repainting_starts: List[float],
    repainting_ends: List[float],
    sample_rate: int = 48000,
    crossfade_duration: float = 0.01,
    layer_gain: float = 1.0,
) -> torch.Tensor:
    """Mix generated Lego layer audio over the original source waveform.

    Args:
        pred_wavs: Generated layer waveform ``[B, C, samples]``.
        src_wavs: Original source waveform ``[B, C, samples]`` or
            ``[C, samples]``.
        repainting_starts: Per-batch layer start time in seconds.
        repainting_ends: Per-batch layer end time in seconds.
        sample_rate: Audio sample rate.
        crossfade_duration: Fade-in/out duration applied only to the layer.
        layer_gain: Linear gain applied to the generated layer before mixing.

    Returns:
        Waveform containing source audio plus the generated layer in the
        selected range.
    """

    pred_wavs = _ensure_batched(pred_wavs)
    src_wavs = _ensure_batched(src_wavs)
    if src_wavs.shape[0] == 1 and pred_wavs.shape[0] > 1:
        src_wavs = src_wavs.expand(pred_wavs.shape[0], -1, -1)

    batch_size = min(pred_wavs.shape[0], src_wavs.shape[0])
    channels = max(pred_wavs.shape[1], src_wavs.shape[1])
    pred_wavs = _match_channels(pred_wavs[:batch_size], channels)
    src_wavs = _match_channels(src_wavs[:batch_size], channels)

    total_samples = max(pred_wavs.shape[-1], src_wavs.shape[-1])
    pred_padded = _pad_to_length(pred_wavs, total_samples)
    src_padded = _pad_to_length(src_wavs, total_samples).to(
        device=pred_padded.device,
        dtype=pred_padded.dtype,
    )
    result = src_padded.clone()

    crossfade_samples = int(crossfade_duration * sample_rate)
    for batch_index in range(batch_size):
        start_sample = int(repainting_starts[batch_index] * sample_rate)
        end_time = float(repainting_ends[batch_index])
        end_sample = total_samples if end_time < 0 else int(end_time * sample_rate)
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        mask = _build_waveform_crossfade_mask(
            total_samples,
            start_sample,
            end_sample,
            crossfade_samples,
            device=pred_padded.device,
        )
        layer = pred_padded[batch_index] * float(layer_gain)
        result[batch_index] = result[batch_index] + layer * mask.unsqueeze(0)

    return result


def _ensure_batched(audio: torch.Tensor) -> torch.Tensor:
    """Return audio with a batch dimension."""

    if audio.dim() == 2:
        return audio.unsqueeze(0)
    return audio


def _match_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    """Return audio with the requested number of channels."""

    current_channels = audio.shape[1]
    if current_channels == channels:
        return audio
    if current_channels == 1:
        return audio.expand(audio.shape[0], channels, audio.shape[-1])
    if current_channels > channels:
        return audio[:, :channels, :]
    pad = torch.zeros(
        audio.shape[0],
        channels - current_channels,
        audio.shape[-1],
        device=audio.device,
        dtype=audio.dtype,
    )
    return torch.cat([audio, pad], dim=1)


def _pad_to_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    """Pad trailing audio samples to ``length``."""

    if audio.shape[-1] >= length:
        return audio[..., :length]
    pad_shape = (*audio.shape[:-1], length - audio.shape[-1])
    pad = torch.zeros(pad_shape, device=audio.device, dtype=audio.dtype)
    return torch.cat([audio, pad], dim=-1)
