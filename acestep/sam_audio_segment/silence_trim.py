"""Compatibility imports for SAM-Audio silence trimming."""

from acestep.audio_processing.silence_trim import (
    SilenceTrimResult,
    audio_sample_count,
    trim_silent_edges,
)

__all__ = ["SilenceTrimResult", "audio_sample_count", "trim_silent_edges"]
