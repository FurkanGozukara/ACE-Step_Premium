"""Remaining-audio helpers for ACE-Step Extract results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from acestep.audio_processing.media_io import read_media_audio
from acestep.audio_utils import save_audio
from acestep.ui.gradio.events.generation.audio_format_options import (
    normalize_extract_audio_format,
)
from acestep.ui.gradio.events.results.extract_remaining_spectral import build_remaining_audio


def save_extract_remaining_audio(
    *,
    source_audio_path: str | Path | None,
    extracted_audio: torch.Tensor,
    sample_rate: int,
    output_dir: str | Path,
    output_stem: str,
    output_format: str,
    mp3_bitrate: str | None = None,
    mp3_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Save source-minus-extracted audio for an ACE-Step Extract run.

    Args:
        source_audio_path: Original source audio or video path.
        extracted_audio: Extracted stem tensor in channel-first layout.
        sample_rate: Extracted tensor sample rate.
        output_dir: Directory where the remaining-audio file should be written.
        output_stem: Base filename stem for the generated sample.
        output_format: Target format: mp3 or wav.
        mp3_bitrate: Optional MP3 bitrate override.
        mp3_sample_rate: Optional MP3 sample-rate override.

    Returns:
        JSON-safe metadata containing the saved path, or an error when skipped.
    """

    if not source_audio_path:
        return {"applied": False, "error": "missing_source_audio"}

    try:
        source_audio, source_sample_rate = read_media_audio(source_audio_path)
        source_tensor = _source_array_to_tensor(source_audio)
        extracted_tensor = extracted_audio.detach().float().cpu()
        source_tensor = _resample_if_needed(
            source_tensor,
            int(source_sample_rate),
            int(sample_rate),
        )
        source_tensor = torch.clamp(source_tensor, min=-1.0, max=1.0)
        extracted_tensor = torch.clamp(extracted_tensor, min=-1.0, max=1.0)
        source_tensor, extracted_tensor = _match_audio_shape(source_tensor, extracted_tensor)
        remaining, subtraction_metadata = build_remaining_audio(source_tensor, extracted_tensor)
        target_format = normalize_extract_audio_format(output_format)
        target_path = Path(output_dir).expanduser().resolve() / (
            f"{output_stem}_remaining.{target_format}"
        )
        saved_path = save_audio(
            remaining,
            target_path,
            sample_rate=int(sample_rate),
            format=target_format,
            channels_first=True,
            mp3_bitrate=mp3_bitrate,
            mp3_sample_rate=mp3_sample_rate,
        ).replace("\\", "/")
    except Exception as exc:
        logger.warning("[extract_remaining] Could not save remaining audio: {}", exc)
        return {"applied": False, "error": str(exc)}

    return {
        "applied": True,
        "remaining_audio_path": saved_path,
        "audio_format": target_format,
        "sample_rate": int(sample_rate),
        "source_sample_rate": int(source_sample_rate),
        "subtraction": subtraction_metadata,
    }


def _source_array_to_tensor(audio: np.ndarray) -> torch.Tensor:
    """Return a channel-first float32 tensor from channel-last source audio."""

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return torch.from_numpy(array[None, :]).float()
    return torch.from_numpy(array.T).float()


def _resample_if_needed(
    audio: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Return audio resampled to the target sample rate when needed."""

    if source_sample_rate == target_sample_rate:
        return audio
    import torchaudio

    return torchaudio.functional.resample(audio, source_sample_rate, target_sample_rate)


def _match_audio_shape(
    source: torch.Tensor,
    extracted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensors with matching channel count and sample length."""

    channels = max(int(source.shape[0]), int(extracted.shape[0]))
    source = _match_channels(source, channels)
    extracted = _match_channels(extracted, channels)
    samples = max(int(source.shape[-1]), int(extracted.shape[-1]))
    return _pad_samples(source, samples), _pad_samples(extracted, samples)


def _match_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    """Return audio with exactly ``channels`` channels."""

    current = int(audio.shape[0])
    if current == channels:
        return audio
    if current == 1:
        return audio.repeat(channels, 1)
    if current > channels:
        return audio[:channels]
    repeats = [audio, audio[-1:].repeat(channels - current, 1)]
    return torch.cat(repeats, dim=0)


def _pad_samples(audio: torch.Tensor, samples: int) -> torch.Tensor:
    """Pad or trim audio to a fixed sample length."""

    current = int(audio.shape[-1])
    if current == samples:
        return audio
    if current > samples:
        return audio[..., :samples]
    return torch.nn.functional.pad(audio, (0, samples - current))
