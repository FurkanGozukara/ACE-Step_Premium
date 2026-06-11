"""Audio output-format options shared by Gradio generation controls."""

from __future__ import annotations


DUAL_AUDIO_FORMAT = "flac_mp3"
DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_MP3_BITRATE = "256k"
DEFAULT_EXTRACT_AUDIO_FORMAT = "mp3"

AUDIO_FORMAT_CHOICES = [
    ("MP3", "mp3"),
    ("WAV", "wav"),
]

SUPPORTED_AUDIO_FORMATS = {
    "mp3",
    "wav",
}

MP3_BITRATE_CHOICES = [
    ("128 kbps", "128k"),
    ("192 kbps", "192k"),
    ("256 kbps", "256k"),
    ("320 kbps", "320k"),
]

MP3_SAMPLE_RATE_CHOICES = [("48 kHz", 48000), ("44.1 kHz", 44100)]

EXTRACT_AUDIO_FORMAT_CHOICES = [
    ("MP3", "mp3"),
    ("WAV", "wav"),
]

SUPPORTED_EXTRACT_AUDIO_FORMATS = {"mp3", "wav"}


def normalize_audio_format(value: object) -> str:
    """Return a supported UI audio-format value."""

    normalized = str(value or "").strip().lower()
    return normalized if normalized in SUPPORTED_AUDIO_FORMATS else DEFAULT_AUDIO_FORMAT


def normalize_extract_audio_format(value: object) -> str:
    """Return a supported ACE-Step Extract output-format value."""

    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_EXTRACT_AUDIO_FORMATS:
        return normalized
    return DEFAULT_EXTRACT_AUDIO_FORMAT


def output_audio_formats(value: object) -> list[str]:
    """Return concrete audio formats that should be written for a UI selection."""

    return [normalize_audio_format(value)]


def primary_audio_format(value: object) -> str:
    """Return the format used for primary playback and backend metadata."""

    return output_audio_formats(value)[0]


def audio_file_extension(value: object) -> str:
    """Return the file extension for a concrete audio format."""

    return normalize_audio_format(value)


def mp3_controls_visible(value: object) -> bool:
    """Return whether MP3 bitrate controls should be visible."""

    return normalize_audio_format(value) == "mp3"


def audio_format_label(value: object) -> str:
    """Return a display label for a UI audio-format value."""

    return normalize_audio_format(value).upper()
