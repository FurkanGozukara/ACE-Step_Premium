"""Compatibility facade for auto-editor based audio trimming."""

from __future__ import annotations

from .auto_editor_trim import SilenceTrimResult, audio_sample_count, trim_silent_edges
from .auto_editor_trim_settings import (
    AUTO_EDITOR_THRESHOLD_DEFAULT_DB as TRIM_THRESHOLD_DEFAULT_DB,
    AUTO_EDITOR_THRESHOLD_MAX_DB as TRIM_THRESHOLD_MAX_DB,
    AUTO_EDITOR_THRESHOLD_MIN_DB as TRIM_THRESHOLD_MIN_DB,
    coerce_auto_editor_threshold_db as clamp_trim_threshold_db,
)

__all__ = [
    "SilenceTrimResult",
    "TRIM_THRESHOLD_DEFAULT_DB",
    "TRIM_THRESHOLD_MAX_DB",
    "TRIM_THRESHOLD_MIN_DB",
    "audio_sample_count",
    "clamp_trim_threshold_db",
    "trim_silent_edges",
]
