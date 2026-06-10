"""FFmpeg trim execution for source range preview media."""

from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from acestep.audio_processing.media_io import is_video_file


SOURCE_RANGE_PREVIEW_TIMEOUT_SECONDS = 60.0


def trim_source_range_preview(source_path: str, start: float, duration: float) -> str:
    """Create or reuse a temporary media file for the selected preview range.

    Args:
        source_path: Source audio or video path.
        start: Range start time in seconds.
        duration: Range duration in seconds.

    Returns:
        Temporary preview media path.
    """

    source = Path(source_path).expanduser().resolve()
    signature = _source_signature(source)
    return _cached_trim_source_range_preview(
        str(source),
        signature[0],
        signature[1],
        start,
        duration,
        is_video_file(source),
    )


def _source_signature(source: Path) -> tuple[int, int]:
    """Return file metadata used to invalidate cached preview trims."""

    try:
        stat = source.stat()
    except OSError:
        return 0, 0
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=64)
def _cached_trim_source_range_preview(
    source_path: str,
    source_mtime_ns: int,
    source_size: int,
    start: float,
    duration: float,
    source_is_video: bool,
) -> str:
    """Trim a source preview range and cache identical preview requests."""

    _ = (source_mtime_ns, source_size)
    source = Path(source_path)
    target_dir = Path(tempfile.mkdtemp(prefix="acestep_range_preview_"))
    if source_is_video:
        return _trim_video_preview(source, target_dir, start, duration)
    return _trim_audio_preview(source, target_dir, start, duration)


def _trim_audio_preview(source: Path, target_dir: Path, start: float, duration: float) -> str:
    """Trim an audio-purpose preview to a temporary WAV file."""

    target_path = target_dir / f"{source.stem or 'source'}_range_preview.wav"
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
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        str(target_path),
    ]
    _run_ffmpeg(cmd, "audio range preview trim failed")
    return str(target_path).replace("\\", "/")


def _trim_video_preview(source: Path, target_dir: Path, start: float, duration: float) -> str:
    """Trim a video preview range to a temporary MP4 file."""

    target_path = target_dir / f"{source.stem or 'source'}_range_preview.mp4"
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
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    _run_ffmpeg(cmd, "video range preview trim failed")
    return str(target_path).replace("\\", "/")


def _run_ffmpeg(cmd: list[str], message: str) -> None:
    """Run an ffmpeg command and raise a compact runtime error on failure."""

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=SOURCE_RANGE_PREVIEW_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{message}: timed out.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"{message}: {stderr}") from exc
