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
    output_format: str = "wav",
    mp3_bitrate: str | None = None,
    mp3_sample_rate: int | None = None,
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
        output_format: Selected clip output format, ``mp3`` or ``wav``.
        mp3_bitrate: Optional MP3 bitrate override.
        mp3_sample_rate: Optional MP3 sample-rate override.

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
    clip_format = _clip_output_format(output_format)
    generated_target = target_dir / f"{key}_latest_repainted_area.{clip_format}"
    original_target = target_dir / f"{key}_latest_repainted_area_original.{clip_format}"
    try:
        generated_clip = _extract_audio_segment(
            generated,
            generated_target,
            segment,
            output_format=clip_format,
            mp3_bitrate=mp3_bitrate,
            mp3_sample_rate=mp3_sample_rate,
        )
        original_clip = _extract_audio_segment(
            source,
            original_target,
            segment,
            output_format=clip_format,
            mp3_bitrate=mp3_bitrate,
            mp3_sample_rate=mp3_sample_rate,
        )
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


def _clip_output_format(value: str) -> str:
    """Return the supported edited-area clip format."""

    return "mp3" if str(value or "").strip().lower() == "mp3" else "wav"


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
    *,
    output_format: str,
    mp3_bitrate: str | None,
    mp3_sample_rate: int | None,
) -> str:
    """Extract an audio segment and return the normalized path."""

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
    cmd.extend(_ffmpeg_audio_output_args(output_format, mp3_bitrate, mp3_sample_rate))
    cmd.append(str(target))
    _run_ffmpeg(cmd)
    return str(target.resolve()).replace("\\", "/")


def _ffmpeg_audio_output_args(
    output_format: str,
    mp3_bitrate: str | None,
    mp3_sample_rate: int | None,
) -> list[str]:
    """Return ffmpeg audio arguments for edited-area clip output."""

    if output_format == "mp3":
        bitrate = str(mp3_bitrate or "256k").strip().lower()
        if bitrate not in {"128k", "192k", "256k", "320k"}:
            bitrate = "256k"
        try:
            sample_rate = int(mp3_sample_rate or 48000)
        except (TypeError, ValueError):
            sample_rate = 48000
        if sample_rate not in {44100, 48000}:
            sample_rate = 48000
        return ["-vn", "-acodec", "libmp3lame", "-b:a", bitrate, "-ar", str(sample_rate)]
    return ["-vn", "-acodec", "pcm_s16le", "-ar", "48000"]


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
