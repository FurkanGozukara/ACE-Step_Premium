"""Metadata duration probing for audio/video files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


FFPROBE_TIMEOUT_SECONDS = 15


def probe_media_duration_seconds(source: Path) -> float:
    """Probe audio stream duration via ffprobe metadata with a short timeout.

    Args:
        source: Media file path to inspect.

    Returns:
        Positive duration in seconds, or ``0.0`` when no usable duration exists.
    """

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe executable was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffprobe media duration probe timed out.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or str(exc)
        raise RuntimeError(f"ffprobe media duration probe failed: {stderr}") from exc

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe media duration probe returned invalid JSON.") from exc

    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "audio":
            duration = _parse_duration_seconds(stream.get("duration"))
            if duration > 0:
                return duration
    return _parse_duration_seconds(payload.get("format", {}).get("duration"))


def _parse_duration_seconds(value: Any) -> float:
    """Parse a positive finite duration value from ffprobe output."""

    try:
        duration = float(value)
    except (TypeError, ValueError):
        return 0.0
    if duration <= 0 or duration != duration or duration in (float("inf"), float("-inf")):
        return 0.0
    return duration
