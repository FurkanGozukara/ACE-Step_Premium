"""Progress reporting helpers for SAM-Audio processing."""

from __future__ import annotations

import json
from typing import Callable

from loguru import logger

ProgressCallback = Callable[[float, str], None]

PROGRESS_PREFIX = "ACESTEP_SAM_AUDIO_PROGRESS "


def report_progress(
    callback: ProgressCallback | None,
    fraction: float,
    message: str,
) -> None:
    """Report clamped progress to an optional callback."""

    fraction = _clamp_fraction(fraction)
    logger.info("[sam_audio] {:>3.0f}% - {}", fraction * 100, message)
    if callback is None:
        return
    callback(fraction, str(message))


def encode_progress_line(fraction: float, message: str) -> str:
    """Encode one subprocess progress event as a prefixed JSON line."""

    return PROGRESS_PREFIX + json.dumps(
        {"progress": _clamp_fraction(fraction), "message": str(message)},
        ensure_ascii=True,
    )


def parse_progress_line(line: str) -> tuple[float, str] | None:
    """Parse one subprocess progress line if it contains a SAM-Audio event."""

    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX):])
        return _clamp_fraction(float(payload["progress"])), str(payload["message"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _clamp_fraction(value: float) -> float:
    """Return a finite progress fraction in the inclusive 0..1 range."""

    if value != value:
        return 0.0
    return max(0.0, min(1.0, float(value)))
