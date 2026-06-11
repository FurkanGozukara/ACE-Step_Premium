"""Waveform insertion for locally generated repaint replacement segments."""

import torch


def apply_repaint_segment_splice(
    pred_segments: torch.Tensor,
    src_wavs: torch.Tensor,
    repainting_starts: list[float],
    repainting_ends: list[float],
    sample_rate: int = 48000,
    crossfade_duration: float = 0.0,
    replacement_strength: float = 1.0,
) -> torch.Tensor:
    """Insert locally generated repaint segments into the original waveform.

    Args:
        pred_segments: Generated replacement audio ``[B, C, segment_samples]``.
        src_wavs: Original source waveform ``[B, C, samples]`` or ``[C, samples]``.
        repainting_starts: Per-batch insertion start time in seconds.
        repainting_ends: Per-batch insertion end time in seconds.
        sample_rate: Audio sample rate.
        crossfade_duration: Optional in-region fade length at both boundaries.
        replacement_strength: Generated-audio mix in the replacement segment.
            ``0.0`` preserves source audio; ``1.0`` uses the generated audio.

    Returns:
        Source waveform with the selected region replaced by the generated audible
        segment, shrinking or expanding output to match the replacement length.
    """
    if src_wavs.dim() == 2:
        src_wavs = src_wavs.unsqueeze(0)
    if src_wavs.shape[0] == 1 and pred_segments.shape[0] > 1:
        src_wavs = src_wavs.expand(pred_segments.shape[0], -1, -1)

    batch_size = min(pred_segments.shape[0], src_wavs.shape[0])
    source_batch = src_wavs[:batch_size].to(
        device=pred_segments.device,
        dtype=pred_segments.dtype,
    )
    crossfade_samples = int(crossfade_duration * sample_rate)
    replacement_strength = max(0.0, min(1.0, float(replacement_strength)))
    spliced_wavs = []

    for b in range(batch_size):
        source = source_batch[b]
        start_sample, end_sample = _resolve_segment_bounds(
            repainting_starts[b],
            repainting_ends[b],
            source.shape[-1],
            sample_rate,
        )
        target_samples = end_sample - start_sample
        if target_samples <= 0:
            spliced_wavs.append(source.clone())
            continue

        segment = _prepare_replacement_segment(
            pred_segments[b],
            target_channels=source.shape[0],
            sample_rate=sample_rate,
        )
        segment = _mix_segment_with_source(
            segment,
            source[:, start_sample:end_sample],
            replacement_strength,
        )
        segment = _apply_segment_edge_fades(segment, crossfade_samples)
        spliced_wavs.append(
            torch.cat(
                [source[:, :start_sample], segment, source[:, end_sample:]],
                dim=-1,
            )
        )

    return _stack_spliced_wavs(spliced_wavs)


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


def _prepare_replacement_segment(
    segment: torch.Tensor,
    target_channels: int,
    sample_rate: int,
) -> torch.Tensor:
    """Return audible segment audio with the requested channel count."""
    if segment.shape[0] != target_channels:
        if segment.shape[0] == 1 and target_channels == 2:
            segment = segment.expand(2, -1)
        else:
            segment = segment[:target_channels]

    return _trim_trailing_silence(segment, sample_rate=sample_rate)


def _mix_segment_with_source(
    segment: torch.Tensor,
    source_region: torch.Tensor,
    replacement_strength: float,
) -> torch.Tensor:
    """Blend a generated replacement segment with the selected source audio."""
    if replacement_strength >= 1.0 or segment.shape[-1] == 0:
        return segment

    source_mix = _fit_source_region_to_segment(source_region, segment.shape[-1])
    if replacement_strength <= 0.0:
        return source_mix
    return replacement_strength * segment + (1.0 - replacement_strength) * source_mix


def _fit_source_region_to_segment(
    source_region: torch.Tensor,
    target_samples: int,
) -> torch.Tensor:
    """Return source-region audio with exactly ``target_samples`` samples."""
    if source_region.shape[-1] > target_samples:
        return source_region[..., :target_samples]
    if source_region.shape[-1] < target_samples:
        return torch.nn.functional.pad(source_region, (0, target_samples - source_region.shape[-1]))
    return source_region


def _trim_trailing_silence(
    segment: torch.Tensor,
    sample_rate: int,
    silence_rms_threshold: float = 0.005,
    window_seconds: float = 0.25,
) -> torch.Tensor:
    """Remove sustained low-RMS trailing silence from a generated segment."""
    if segment.shape[-1] == 0:
        return segment

    window_samples = max(1, int(sample_rate * window_seconds))
    total_samples = segment.shape[-1]
    end = total_samples

    while end > 0:
        start = max(0, end - window_samples)
        window = segment[:, start:end]
        rms = torch.sqrt(torch.mean(window.float() * window.float()))
        if float(rms.item()) >= silence_rms_threshold:
            return segment[..., :end]
        end = start

    return segment[..., :0]


def _apply_segment_edge_fades(
    segment: torch.Tensor,
    crossfade_samples: int,
) -> torch.Tensor:
    """Apply optional short fades to a variable-length replacement segment."""
    fade_samples = min(crossfade_samples, segment.shape[-1] // 2)
    if fade_samples <= 0:
        return segment

    faded = segment.clone()
    fade_in = torch.linspace(0.0, 1.0, fade_samples + 2, device=segment.device)[1:-1]
    fade_out = torch.linspace(1.0, 0.0, fade_samples + 2, device=segment.device)[1:-1]
    faded[:, :fade_samples] *= fade_in.unsqueeze(0)
    faded[:, -fade_samples:] *= fade_out.unsqueeze(0)
    return faded


def _stack_spliced_wavs(spliced_wavs: list[torch.Tensor]) -> torch.Tensor:
    """Stack variable-length spliced waveforms, padding only for batch shape."""
    max_samples = max(wav.shape[-1] for wav in spliced_wavs)
    padded_wavs = []
    for wav in spliced_wavs:
        if wav.shape[-1] < max_samples:
            wav = torch.nn.functional.pad(wav, (0, max_samples - wav.shape[-1]))
        padded_wavs.append(wav)
    return torch.stack(padded_wavs, dim=0)
