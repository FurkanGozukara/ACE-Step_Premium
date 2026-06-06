"""Coercion helpers for Audio Processing settings."""

from __future__ import annotations

from typing import Any

from .presets import DEFAULT_STAGE_VALUES, STAGE_KEYS


def coerce_output_format(value: Any) -> str:
    """Return a supported processed-audio output format."""

    normalized = str(value or "wav").strip().lower()
    return normalized if normalized in {"wav", "flac", "mp3"} else "wav"


def coerce_stage_values(raw_values: Any) -> dict[str, float]:
    """Return numeric stage values with defaults filled in."""

    source = raw_values if isinstance(raw_values, dict) else {}
    return {
        key: coerce_float_value(source.get(key), DEFAULT_STAGE_VALUES[key])
        for key in STAGE_KEYS
    }


def coerce_stage_enabled(raw_values: Any) -> dict[str, bool]:
    """Return stage enabled flags with defaults filled in."""

    source = raw_values if isinstance(raw_values, dict) else {}
    return {key: bool(source.get(key, True)) for key in STAGE_KEYS}


def coerce_float_value(value: Any, fallback: float) -> float:
    """Return a finite float or a fallback."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if result != result or result in (float("inf"), float("-inf")):
        return fallback
    return result
