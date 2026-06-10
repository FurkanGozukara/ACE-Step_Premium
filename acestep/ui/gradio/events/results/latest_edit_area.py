"""Create latest edited-area comparison clips for inline previews."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger


EDIT_AREA_COMPARE_TASKS = frozenset({"cover", "cover-nofsq", "repaint", "lego", "complete"})
_RANGE_TASKS = frozenset({"repaint", "lego", "complete"})
_FFMPEG_TIMEOUT_SECONDS = 120.0
_MIN_DURATION_SECONDS = 0.05


def create_latest_edit_area_clips(
    *,
    task_type: str | None,
    generated_audio_path: str | None,
    source_audio_path: str | None,
    run_dir: str | Path,
    key: str,
    repainting_start: Any,
    repainting_end: Any,
) -> dict[str, Any]:
    """Create generated/original edited-area clips for source-edit tasks.

    Args:
        task_type: Backend task type.
        generated_audio_path: Saved Sample 1 output path.
        source_audio_path: Saved source/original input path.
        run_dir: Current generation output directory.
        key: Sample key used to name clip files.
        repainting_start: Selected source range start in seconds.
        repainting_end: Selected source range end in seconds, or ``-1`` for source end.

    Returns:
        Metadata with ``applied`` plus generated/original clip paths when available.
    """

    task = str(task_type or "").strip().lower()
    if task not in EDIT_AREA_COMPARE_TASKS:
        return {"applied": False, "reason": "unsupported_task"}

    generated = _existing_file(generated_audio_path)
    source = _existing_file(source_audio_path)
    if generated is None or source is None:
        return {"applied": False, "reason": "missing_audio"}

    segment = _segment_for_task(task, repainting_start, repainting_end)
    if segment is None:
        return {"applied": False, "reason": "invalid_range"}

    target_dir = Path(run_dir)
    generated_target = target_dir / f"{key}_latest_repainted_area.wav"
    original_target = target_dir / f"{key}_latest_repainted_area_original.wav"
    try:
        generated_clip = _extract_audio_segment(generated, generated_target, segment)
        original_clip = _extract_audio_segment(source, original_target, segment)
    except RuntimeError as exc:
        logger.warning(f"Failed to create latest edited-area clips: {exc}")
        return {"applied": False, "reason": "clip_failed", "error": str(exc)}

    return {
        "applied": True,
        "generated_area_path": generated_clip,
        "original_area_path": original_clip,
        "start": segment[0],
        "end": None if segment[1] is None else segment[0] + segment[1],
        "task_type": task,
    }


def _segment_for_task(
    task_type: str,
    repainting_start: Any,
    repainting_end: Any,
) -> tuple[float, float | None] | None:
    """Return ``(start, duration)`` for the inline edited-area clip."""

    if task_type not in _RANGE_TASKS:
        return 0.0, None

    start = _parse_seconds(repainting_start)
    end = _parse_seconds(repainting_end)
    if start is None or end is None or start < 0:
        return None
    if end < 0:
        return round(start, 3), None
    if end <= start or (end - start) < _MIN_DURATION_SECONDS:
        return None
    return round(start, 3), round(end - start, 3)


def _parse_seconds(value: Any) -> float | None:
    """Parse a finite seconds value."""

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return seconds


def _existing_file(path: str | None) -> Path | None:
    """Return an existing media path or ``None``."""

    if not path:
        return None
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError:
        pass
    return candidate if candidate.is_file() else None


def _extract_audio_segment(
    source: Path,
    target: Path,
    segment: tuple[float, float | None],
) -> str:
    """Extract an audio segment to WAV and return the normalized path."""

    start, duration = segment
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
    ]
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "48000", str(target)])
    _run_ffmpeg(cmd)
    return str(target.resolve()).replace("\\", "/")


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg and raise a compact error on failure."""

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("edited-area clip extraction timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"edited-area clip extraction failed: {stderr}") from exc
