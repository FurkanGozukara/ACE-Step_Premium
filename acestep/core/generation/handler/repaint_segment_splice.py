"""Waveform insertion for locally generated repaint replacement segments."""

from typing import List

import torch


def apply_repaint_segment_splice(
    pred_segments: torch.Tensor,
    src_wavs: torch.Tensor,
    repainting_starts: List[float],
    repainting_ends: List[float],
    sample_rate: int = 48000,
    crossfade_duration: float = 0.0,
) -> torch.Tensor:
    """Insert locally generated repaint segments into the original waveform.

    Args:
        pred_segments: Generated replacement audio ``[B, C, segment_samples]``.
        src_wavs: Original source waveform ``[B, C, samples]`` or ``[C, samples]``.
        repainting_starts: Per-batch insertion start time in seconds.
        repainting_ends: Per-batch insertion end time in seconds.
        sample_rate: Audio sample rate.
        crossfade_duration: Optional in-region fade length at both boundaries.

    Returns:
        Full-length source waveform with each generated segment inserted.
    """
    if src_wavs.dim() == 2:
        src_wavs = src_wavs.unsqueeze(0)
    if src_wavs.shape[0] == 1 and pred_segments.shape[0] > 1:
        src_wavs = src_wavs.expand(pred_segments.shape[0], -1, -1)

    batch_size = min(pred_segments.shape[0], src_wavs.shape[0])
    result = src_wavs[:batch_size].to(
        device=pred_segments.device,
        dtype=pred_segments.dtype,
    ).clone()
    crossfade_samples = int(crossfade_duration * sample_rate)

    for b in range(batch_size):
        start_sample, end_sample = _resolve_segment_bounds(
            repainting_starts[b],
            repainting_ends[b],
            result.shape[-1],
            sample_rate,
        )
        target_samples = end_sample - start_sample
        if target_samples <= 0:
            continue

        segment = _fit_segment_to_region(
            pred_segments[b],
            target_channels=result.shape[1],
            target_samples=target_samples,
        )
        existing = result[b, :, start_sample:end_sample]
        mask = _build_segment_crossfade_mask(
            target_samples,
            crossfade_samples,
            pred_segments.device,
        ).unsqueeze(0)
        result[b, :, start_sample:end_sample] = mask * segment + (1.0 - mask) * existing

    return result


def _resolve_segment_bounds(
    repainting_start: float,
    repainting_end: float,
    total_samples: int,
    sample_rate: int,
) -> tuple[int, int]:
    """Clamp repaint times to valid waveform sample bounds."""
    start_sample = int(repainting_start * sample_rate)
    end_sample = int(repainting_end * sample_rate)
    start_sample = max(0, min(start_sample, total_samples))
    end_sample = max(start_sample, min(end_sample, total_samples))
    return start_sample, end_sample


def _fit_segment_to_region(
    segment: torch.Tensor,
    target_channels: int,
    target_samples: int,
) -> torch.Tensor:
    """Return segment audio with the requested channel count and duration."""
    if segment.shape[0] != target_channels:
        if segment.shape[0] == 1 and target_channels == 2:
            segment = segment.expand(2, -1)
        else:
            segment = segment[:target_channels]

    if segment.shape[-1] > target_samples:
        return segment[..., :target_samples]
    if segment.shape[-1] < target_samples:
        return torch.nn.functional.pad(segment, (0, target_samples - segment.shape[-1]))
    return segment


def _build_segment_crossfade_mask(
    total_samples: int,
    crossfade_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a local replacement mask with optional fades inside the segment."""
    mask = torch.ones(total_samples, device=device)
    fade_samples = min(crossfade_samples, total_samples // 2)
    if fade_samples <= 0:
        return mask

    fade_in = torch.linspace(0.0, 1.0, fade_samples + 2, device=device)[1:-1]
    fade_out = torch.linspace(1.0, 0.0, fade_samples + 2, device=device)[1:-1]
    mask[:fade_samples] = fade_in
    mask[-fade_samples:] = torch.minimum(mask[-fade_samples:], fade_out)
    return mask
