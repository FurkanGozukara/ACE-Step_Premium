"""Prompt normalization helpers for Lego track generation."""

from __future__ import annotations

import re


VOCAL_LEGO_TRACKS = {"vocals", "backing_vocals"}


def extract_lego_track_name(instruction: str | None) -> str:
    """Extract the target Lego track name from a task instruction.

    Args:
        instruction: Instruction text such as
            ``"Generate the GUITAR track based on the audio context:"``.

    Returns:
        Lowercase normalized track name, or an empty string when unavailable.
    """

    text = str(instruction or "")
    match = re.search(r"generate\s+the\s+(.+?)\s+track\b", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_lego_lyrics(
    task_type: str | None,
    instruction: str | None,
    lyrics: str | None,
) -> str:
    """Return lyrics suitable for the requested Lego track.

    Non-vocal Lego tracks should generate an instrumental layer. Passing full
    song lyrics to a guitar/drums/bass request strongly biases the lyric branch
    toward vocals, so these tracks use an instrumental lyric condition.

    Args:
        task_type: Backend task type.
        instruction: Task instruction containing the target track name.
        lyrics: User-provided lyric text.

    Returns:
        Original lyrics for non-Lego or vocal Lego tracks; otherwise
        ``"[Instrumental]"`` for a known non-vocal Lego target.
    """

    if str(task_type or "").strip().lower() != "lego":
        return lyrics or ""
    track_name = extract_lego_track_name(instruction)
    if not track_name or track_name in VOCAL_LEGO_TRACKS:
        return lyrics or ""
    return "[Instrumental]"
