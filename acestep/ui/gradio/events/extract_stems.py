"""Helpers for ACE-Step Extract stem names and output suffixes."""

from __future__ import annotations

from acestep.constants import TRACK_NAMES

_TRACK_NAME_LOOKUP = {name.lower(): name for name in TRACK_NAMES}
_TRACK_NAME_ALIASES = {
    "vocal": "vocals",
    "backing_vocal": "backing_vocals",
    "backingvocal": "backing_vocals",
    "backingvocals": "backing_vocals",
}
_STEM_FILENAME_SUFFIXES = {
    "vocals": "vocal",
    "backing_vocals": "backing_vocal",
}


def supported_extract_track_names() -> list[str]:
    """Return supported Extract track names in the model's canonical order."""

    return list(TRACK_NAMES)


def normalize_extract_track_name(track_name: str | None) -> str:
    """Return a canonical supported Extract track name.

    Args:
        track_name: UI or metadata value to normalize.

    Raises:
        ValueError: If the value is empty or not a supported Extract stem.
    """

    raw_value = str(track_name or "").strip()
    if not raw_value:
        raise ValueError("Select Track Name before running Extract.")

    normalized = _normalize_track_token(raw_value)
    normalized = _TRACK_NAME_ALIASES.get(normalized, normalized)
    if normalized in _TRACK_NAME_LOOKUP:
        return _TRACK_NAME_LOOKUP[normalized]

    valid = ", ".join(TRACK_NAMES)
    raise ValueError(f"Unsupported Extract Track Name '{raw_value}'. Choose one of: {valid}.")


def extract_stem_filename_suffix(track_name: str | None) -> str:
    """Return the filesystem suffix for one Extract stem."""

    if track_name is None or str(track_name).strip() == "":
        return ""
    normalized = normalize_extract_track_name(track_name)
    return _STEM_FILENAME_SUFFIXES.get(normalized, normalized)


def _normalize_track_token(value: str) -> str:
    """Normalize a free-form track label into the model's token shape."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in normalized.split("_") if part)
