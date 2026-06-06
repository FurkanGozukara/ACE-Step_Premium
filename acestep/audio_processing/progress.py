"""Progress line encoding for Audio Processing subprocess workers."""

from __future__ import annotations

import json
from typing import Callable


ProgressCallback = Callable[[float | None, str], None]
PROGRESS_PREFIX = "ACESTEP_AUDIO_PROCESSING_PROGRESS "


def encode_progress_line(progress: float | None, message: str) -> str:
    """Encode one subprocess progress event as a prefixed JSON line."""

    return PROGRESS_PREFIX + json.dumps(
        {"progress": _clamp_optional_fraction(progress), "message": str(message)},
        ensure_ascii=True,
    )


def parse_progress_line(line: str) -> tuple[float | None, str] | None:
    """Parse one subprocess progress line if it contains an Audio Processing event."""

    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX):])
        return _clamp_optional_fraction(payload.get("progress")), str(payload["message"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _clamp_optional_fraction(value: object) -> float | None:
    """Return a finite progress fraction or ``None``."""

    if value is None:
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction != fraction:
        return None
    return max(0.0, min(1.0, fraction))
