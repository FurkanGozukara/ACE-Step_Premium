"""Hard-overwrite Remix area splicing for full-song result previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from acestep.audio_processing.media_io import read_media_audio
from acestep.ui.gradio.events.generation.audio_format_options import audio_file_extension
from acestep.ui.gradio.events.results.remix_source_range import (
    resolve_bounded_remix_source_range,
)


def save_remix_area_splice(
    *,
    task_type: str | None,
    generated_audio_path: str | None,
    source_audio_path: str | None,
    run_dir: str | Path,
    key: str,
    repainting_start: Any,
    repainting_end: Any,
    output_format: str = "wav",
    mp3_bitrate: str | None = None,
    mp3_sample_rate: int | None = None,
    save_audio_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Save full source audio with the selected range replaced by Remix output.

    This is Remix-only behavior: the model output is a full-song remix, but when
    the user chooses a bounded Remix Source Start/End range, Sample 1 becomes the
    original song with that range overwritten 100% by the corresponding part of
    the full remix. No crossfade or wet/dry blend is applied.
    """

    source_range = resolve_bounded_remix_source_range(
        task_type,
        source_audio_path,
        repainting_start,
        repainting_end,
    )
    if source_range is None:
        return {"applied": False, "reason": "not_bounded_remix_range"}

    generated = _existing_file(generated_audio_path)
    source = _existing_file(source_audio_path)
    if generated is None or source is None:
        return {"applied": False, "reason": "missing_audio"}

    try:
        source_audio, source_sample_rate = read_media_audio(source)
        generated_audio, generated_sample_rate = read_media_audio(generated)
        merged = _overwrite_source_range(
            source_audio=source_audio,
            generated_audio=generated_audio,
            source_sample_rate=source_sample_rate,
            generated_sample_rate=generated_sample_rate,
            start_seconds=source_range.start,
            duration_seconds=source_range.duration,
        )
    except Exception as exc:
        logger.warning("[remix_area_splice] Failed to prepare Remix splice: {}", exc)
        return {"applied": False, "reason": "splice_failed", "error": str(exc)}

    target_format = _splice_output_format(output_format)
    target = Path(run_dir) / f"{key}_remix_merged.{audio_file_extension(target_format)}"
    save_fn = save_audio_fn or _default_save_audio
    try:
        saved_path = save_fn(
            audio_data=merged.T,
            output_path=str(target),
            sample_rate=int(source_sample_rate),
            format=target_format,
            channels_first=True,
            mp3_bitrate=mp3_bitrate,
            mp3_sample_rate=mp3_sample_rate,
        )
    except Exception as exc:
        logger.warning("[remix_area_splice] Failed to save Remix splice: {}", exc)
        return {"applied": False, "reason": "save_failed", "error": str(exc)}

    audio_path = str(saved_path or target).replace("\\", "/")
    return {
        "applied": True,
        "audio_path": audio_path,
        "full_remix_audio_path": str(generated).replace("\\", "/"),
        "source_audio_path": str(source).replace("\\", "/"),
        "start": source_range.start,
        "end": source_range.start + source_range.duration,
        "output_format": target_format,
    }


def _overwrite_source_range(
    *,
    source_audio: np.ndarray,
    generated_audio: np.ndarray,
    source_sample_rate: int,
    generated_sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    """Return source audio with selected samples replaced by generated audio."""

    source = _ensure_channel_last(source_audio)
    generated = _ensure_channel_last(generated_audio)
    if int(generated_sample_rate) != int(source_sample_rate):
        generated = _resample_linear(
            generated,
            source_rate=int(generated_sample_rate),
            target_rate=int(source_sample_rate),
        )
    generated = _match_channels(generated, source.shape[1])

    start_sample = max(0, int(round(float(start_seconds) * int(source_sample_rate))))
    end_sample = max(
        start_sample,
        int(round((float(start_seconds) + float(duration_seconds)) * int(source_sample_rate))),
    )
    start_sample = min(start_sample, source.shape[0])
    end_sample = min(end_sample, source.shape[0], generated.shape[0])
    if end_sample <= start_sample:
        raise ValueError("selected Remix range does not overlap generated audio")

    merged = source.copy()
    merged[start_sample:end_sample, :] = generated[start_sample:end_sample, :]
    return merged.astype(np.float32, copy=False)


def _ensure_channel_last(audio: np.ndarray) -> np.ndarray:
    """Return a float32 ``[samples, channels]`` audio array."""

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio, got shape {array.shape}")
    return array


def _match_channels(audio: np.ndarray, channel_count: int) -> np.ndarray:
    """Adapt generated audio channels to the source channel count."""

    if audio.shape[1] == channel_count:
        return audio
    if channel_count == 1:
        return audio.mean(axis=1, keepdims=True)
    if audio.shape[1] == 1:
        return np.repeat(audio, channel_count, axis=1)
    if audio.shape[1] > channel_count:
        return audio[:, :channel_count]

    repeats = int(np.ceil(channel_count / audio.shape[1]))
    return np.tile(audio, (1, repeats))[:, :channel_count]


def _resample_linear(audio: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample channel-last audio with a lightweight linear fallback."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return audio
    if audio.shape[0] == 0:
        return audio

    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    old_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    channels = [
        np.interp(new_x, old_x, audio[:, channel]).astype(np.float32)
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1)


def _splice_output_format(value: str | None) -> str:
    """Return a supported Remix splice output format."""

    return "mp3" if str(value or "").strip().lower() == "mp3" else "wav"


def _existing_file(path: str | Path | None) -> Path | None:
    """Return an existing path or ``None``."""

    if not path:
        return None
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError:
        pass
    return candidate if candidate.is_file() else None


def _default_save_audio(**kwargs) -> str:
    """Import the default audio saver only when needed."""

    from acestep.audio_utils import save_audio

    return save_audio(**kwargs)
