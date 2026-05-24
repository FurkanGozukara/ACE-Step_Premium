"""Detect vocal cues in auto-label metadata."""

from __future__ import annotations

import re
from typing import Any


_VOCAL_HINT_RE = re.compile(
    r"\b("
    r"ad[- ]?libs?|"
    r"chants?|"
    r"lyric(?:s|al)?|"
    r"rap(?:ped|per|pers|ping)?|"
    r"sang|"
    r"singer|singers|singing|sung|"
    r"spoken(?:[- ]word)?|"
    r"verse|verses|"
    r"voice|voices|"
    r"vocal|vocals"
    r")\b",
    re.IGNORECASE,
)


def metadata_suggests_vocals(metadata: dict[str, Any] | None) -> bool:
    """Return whether auto-label metadata clearly describes vocal content.

    Args:
        metadata: LM metadata containing caption, genre, or similar text fields.

    Returns:
        True when the text contains clear vocal-performance terms.
    """

    if not metadata:
        return False

    text_parts = [
        str(metadata.get("caption") or ""),
        str(metadata.get("genre") or ""),
        str(metadata.get("genres") or ""),
    ]
    return bool(_VOCAL_HINT_RE.search(" ".join(text_parts)))


def vocal_without_lyrics_status(rejection_reason: str | None = "") -> str:
    """Return a status suffix for vocal tracks without usable lyric text."""

    if rejection_reason:
        return f"(vocal metadata; LM transcription rejected: {rejection_reason})"
    return "(vocal metadata; no usable lyrics transcribed)"
