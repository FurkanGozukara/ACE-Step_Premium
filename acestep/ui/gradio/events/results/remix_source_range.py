"""Resolve Remix source-audio range selection for post-generation splicing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


REMIX_SOURCE_RANGE_TASKS = frozenset({"cover", "cover-nofsq"})
_MIN_RANGE_SECONDS = 0.05


@dataclass(frozen=True)
class RemixSourceRange:
    """Validated Remix source segment in seconds."""

    start: float
    duration: float
    source_duration: float | None


def resolve_remix_source_range_audio(
    task_type: str | None,
    source_audio_path: str | None,
    repainting_start: Any,
    repainting_end: Any,
) -> str | None:
    """Return the full source path used for Remix generation.

    Remix ranges no longer trim the source before generation. The model remixes
    the whole source, then the selected range is spliced back into the original
    song after generation.
    """

    _ = task_type, repainting_start, repainting_end
    return source_audio_path


def resolve_bounded_remix_source_range(
    task_type: str | None,
    source_audio_path: str | None,
    repainting_start: Any,
    repainting_end: Any,
) -> RemixSourceRange | None:
    """Return a valid non-full Remix range for post-generation replacement."""

    if str(task_type or "").strip().lower() not in REMIX_SOURCE_RANGE_TASKS:
        return None
    if not source_audio_path:
        return None

    source = Path(str(source_audio_path)).expanduser()
    if not source.is_file():
        return None

    source_range = _validated_remix_source_range(
        str(source),
        repainting_start,
        repainting_end,
    )
    if source_range is None or _covers_full_source(source_range):
        return None
    return source_range


def remix_source_segment_for_clips(
    task_type: str | None,
    source_audio_path: str | None,
    repainting_start: Any,
    repainting_end: Any,
) -> tuple[float, float | None]:
    """Return the generated/original comparison segment for Remix previews.

    Bounded ranges return that selected area. Missing, invalid, or full-source
    ranges fall back to the whole Remix/source comparison.
    """

    source_range = resolve_bounded_remix_source_range(
        task_type,
        source_audio_path,
        repainting_start,
        repainting_end,
    )
    if source_range is None:
        return 0.0, None
    return source_range.start, source_range.duration


def _validated_remix_source_range(
    source_path: str,
    repainting_start: Any,
    repainting_end: Any,
) -> RemixSourceRange | None:
    """Return a valid Remix range, or ``None`` for invalid in-progress values."""

    start = _parse_seconds(repainting_start)
    end = _parse_seconds(repainting_end)
    if start is None or end is None or start < 0:
        return None

    source_duration = _safe_source_duration_seconds(source_path)
    if source_duration is not None and source_duration <= start:
        return None

    if end < 0:
        if source_duration is None:
            return None
        actual_end = source_duration
    else:
        if end <= start:
            return None
        actual_end = min(end, source_duration) if source_duration is not None else end

    duration = actual_end - start
    if duration < _MIN_RANGE_SECONDS:
        return None
    return RemixSourceRange(
        start=round(start, 3),
        duration=round(duration, 3),
        source_duration=source_duration,
    )


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
        duration = float(_media_audio_duration_seconds(source_path))
    except Exception as exc:
        logger.debug(f"Skipping Remix source range clamp for {source_path}: {exc}")
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _covers_full_source(source_range: RemixSourceRange) -> bool:
    """Return whether the selected range is effectively the whole source."""

    if source_range.source_duration is None:
        return False
    return source_range.start <= 0.0 and (
        source_range.duration >= source_range.source_duration - _MIN_RANGE_SECONDS
    )


def _media_audio_duration_seconds(source_path: str) -> float:
    """Import and call media duration probing only when Remix range trimming needs it."""

    from acestep.audio_processing.media_io import media_audio_duration_seconds

    return media_audio_duration_seconds(source_path)


def _trim_source_range_preview(source_path: str, start: float, duration: float) -> str:
    """Deprecated compatibility shim; Remix no longer trims before generation."""

    _ = start, duration
    return source_path
