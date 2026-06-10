"""Source range preview helpers for generation start/end controls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
from loguru import logger

from acestep.audio_processing.media_io import is_video_file, media_audio_duration_seconds
from acestep.ui.gradio.media_upload_values import (
    latest_upload_path,
    resolve_effective_source_audio,
)
from .media_range_preview_trim import trim_source_range_preview


SOURCE_RANGE_PREVIEW_MAX_SECONDS = 60.0
SOURCE_RANGE_PREVIEW_MODES = frozenset(
    {"Custom", "Remix", "Repaint", "Extract", "Lego", "Complete"}
)
_MIN_RANGE_SECONDS = 0.05


@dataclass(frozen=True)
class SourceRangePreview:
    """Validated source preview range in seconds."""

    start: float
    duration: float


def preview_source_range(
    src_audio: Any,
    src_audio_preview: Any,
    src_audio_preview_original: Any,
    repainting_start: Any,
    repainting_end: Any,
    mode: str,
) -> tuple[Any, Any]:
    """Return audio/video preview updates for the selected source range.

    Args:
        src_audio: Original Source Audio upload value.
        src_audio_preview: Current editable Source Audio Preview value.
        src_audio_preview_original: Preview path created from the original upload.
        repainting_start: Selected range start in seconds.
        repainting_end: Selected range end in seconds, or ``-1`` for source end.
        mode: Current generation mode.

    Returns:
        Tuple of ``(audio_preview_update, video_preview_update)``.
    """

    if mode not in SOURCE_RANGE_PREVIEW_MODES:
        return _hide_audio(), _hide_video()

    source_path = resolve_effective_source_audio(
        src_audio,
        src_audio_preview,
        src_audio_preview_original,
    )
    source_path = latest_upload_path(source_path)
    if not source_path or not Path(source_path).is_file():
        return _hide_audio(), _hide_video()

    preview_range = _validated_preview_range(source_path, repainting_start, repainting_end)
    if preview_range is None:
        return _hide_audio(), _hide_video()

    try:
        preview_path = trim_source_range_preview(
            source_path,
            preview_range.start,
            preview_range.duration,
        )
    except RuntimeError as exc:
        logger.warning(f"Failed to create source range preview: {exc}")
        return _hide_audio(), _hide_video()

    if is_video_file(source_path) and Path(preview_path).suffix.lower() != ".wav":
        return _hide_audio(), gr.update(value=preview_path, visible=True)
    return gr.update(value=preview_path, visible=True), _hide_video()


def _validated_preview_range(
    source_path: str,
    repainting_start: Any,
    repainting_end: Any,
) -> SourceRangePreview | None:
    """Return a safe preview range, or ``None`` for invalid in-progress input."""

    start = _parse_seconds(repainting_start)
    end = _parse_seconds(repainting_end)
    if start is None or end is None or start < 0:
        return None

    source_duration = _safe_source_duration_seconds(source_path)
    if source_duration is not None and source_duration <= start:
        return None

    if end < 0:
        duration = (
            source_duration - start
            if source_duration is not None
            else SOURCE_RANGE_PREVIEW_MAX_SECONDS
        )
    else:
        if end <= start:
            return None
        actual_end = min(end, source_duration) if source_duration is not None else end
        duration = actual_end - start

    duration = min(duration, SOURCE_RANGE_PREVIEW_MAX_SECONDS)
    if duration < _MIN_RANGE_SECONDS:
        return None
    return SourceRangePreview(start=round(start, 3), duration=round(duration, 3))


def _parse_seconds(value: Any) -> float | None:
    """Parse a finite seconds value from a Gradio number value."""

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return seconds


def _safe_source_duration_seconds(source_path: str) -> float | None:
    """Return source duration when available without surfacing probe failures."""

    try:
        duration = float(media_audio_duration_seconds(source_path))
    except Exception as exc:
        logger.debug(f"Skipping source range duration clamp for {source_path}: {exc}")
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _hide_audio() -> Any:
    """Return a hidden empty audio update."""

    return gr.update(value=None, visible=False)


def _hide_video() -> Any:
    """Return a hidden empty video update."""

    return gr.update(value=None, visible=False)
