"""Settings for auto-editor based silence trimming."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


AUTO_EDITOR_THRESHOLD_DEFAULT_DB = -40.0
AUTO_EDITOR_THRESHOLD_MIN_DB = -100.0
AUTO_EDITOR_THRESHOLD_MAX_DB = 0.0
AUTO_EDITOR_MARGIN_DEFAULT_SECONDS = 0.3
AUTO_EDITOR_MARGIN_MIN_SECONDS = 0.0
AUTO_EDITOR_MARGIN_MAX_SECONDS = 5.0
AUTO_EDITOR_MINCUT_DEFAULT = 20
AUTO_EDITOR_MINCLIP_DEFAULT = 4
AUTO_EDITOR_SMOOTH_MIN = 0
AUTO_EDITOR_SMOOTH_MAX = 300
AUTO_EDITOR_ANALYSIS_I = -24
AUTO_EDITOR_ANALYSIS_TP = -2
AUTO_EDITOR_ANALYSIS_LRA = 7
AUTO_EDITOR_ANALYSIS_SAMPLE_RATE = 48000
AUTO_EDITOR_ANALYSIS_CHANNELS = 2


@dataclass(frozen=True)
class AutoEditorTrimSettings:
    """Auto-editor trim options used to detect and cut inactive audio."""

    threshold_db: float = AUTO_EDITOR_THRESHOLD_DEFAULT_DB
    margin_seconds: float = AUTO_EDITOR_MARGIN_DEFAULT_SECONDS
    mincut: int = AUTO_EDITOR_MINCUT_DEFAULT
    minclip: int = AUTO_EDITOR_MINCLIP_DEFAULT
    normalize_analysis_audio: bool = True

    def to_payload(self) -> dict[str, object]:
        """Return JSON-safe settings values."""

        return {
            "threshold_db": self.threshold_db,
            "margin_seconds": self.margin_seconds,
            "mincut": self.mincut,
            "minclip": self.minclip,
            "normalize_analysis_audio": self.normalize_analysis_audio,
        }


DEFAULT_AUTO_EDITOR_TRIM_SETTINGS = AutoEditorTrimSettings()


def auto_editor_trim_settings_from_payload(payload: Any) -> AutoEditorTrimSettings:
    """Build trim settings from a dict-like payload."""

    source = payload if isinstance(payload, dict) else {}
    return AutoEditorTrimSettings(
        threshold_db=coerce_auto_editor_threshold_db(source.get("threshold_db")),
        margin_seconds=coerce_auto_editor_margin_seconds(source.get("margin_seconds")),
        mincut=coerce_auto_editor_smooth_value(source.get("mincut"), AUTO_EDITOR_MINCUT_DEFAULT),
        minclip=coerce_auto_editor_smooth_value(
            source.get("minclip"),
            AUTO_EDITOR_MINCLIP_DEFAULT,
        ),
        normalize_analysis_audio=bool(source.get("normalize_analysis_audio", True)),
    )


def coerce_auto_editor_threshold_db(value: Any) -> float:
    """Return a finite auto-editor threshold in the supported dB range."""

    number = _finite_float(value, AUTO_EDITOR_THRESHOLD_DEFAULT_DB)
    return max(AUTO_EDITOR_THRESHOLD_MIN_DB, min(AUTO_EDITOR_THRESHOLD_MAX_DB, number))


def coerce_auto_editor_margin_seconds(value: Any) -> float:
    """Return a finite auto-editor margin value in seconds."""

    number = _finite_float(value, AUTO_EDITOR_MARGIN_DEFAULT_SECONDS)
    return max(AUTO_EDITOR_MARGIN_MIN_SECONDS, min(AUTO_EDITOR_MARGIN_MAX_SECONDS, number))


def coerce_auto_editor_smooth_value(value: Any, fallback: int) -> int:
    """Return a finite auto-editor smooth value."""

    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = int(fallback)
    return max(AUTO_EDITOR_SMOOTH_MIN, min(AUTO_EDITOR_SMOOTH_MAX, number))


def _finite_float(value: Any, fallback: float) -> float:
    """Return a finite float or fallback."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback
