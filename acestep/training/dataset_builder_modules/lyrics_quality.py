"""Quality checks for LM-formatted training lyrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from .lyrics_cleanup import clean_raw_lyrics_for_training


_SECTION_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_MIN_REPETITION_LINES = 12
_MIN_WORDS_FOR_REPETITION = 24
_MAX_TOP_LINE_SHARE = 0.35
_MAX_TOP_WORD_SHARE = 0.30
_MIN_UNIQUE_LINE_SHARE = 0.35
_MAX_WORD_RUN = 8
_MIN_RAW_WORDS_FOR_OVERLAP = 12
_MIN_FORMATTED_WORDS_FOR_OVERLAP = 8
_MIN_FORMATTED_RAW_OVERLAP = 0.25
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "up",
    "with",
}


@dataclass(frozen=True)
class LyricsSelection:
    """Selected training lyrics and optional rejected formatted text reason."""

    lyrics: str
    formatted_lyrics: str
    rejection_reason: str = ""


def select_training_lyrics(raw_lyrics: str, formatted_lyrics: str) -> LyricsSelection:
    """Choose formatted lyrics only when they remain useful for training.

    Args:
        raw_lyrics: Original lyrics loaded from a sidecar text file.
        formatted_lyrics: Lyrics returned by the LM formatter.

    Returns:
        LyricsSelection containing the lyrics to train on, the formatted lyrics to
        persist, and a rejection reason when the raw lyrics were kept instead.
    """

    raw_lyrics = str(raw_lyrics or "").strip()
    formatted_lyrics = str(formatted_lyrics or "").strip()

    if not formatted_lyrics:
        return LyricsSelection(raw_lyrics, "")

    rejection_reason = get_formatted_lyrics_rejection_reason(raw_lyrics, formatted_lyrics)
    if rejection_reason:
        fallback_lyrics = clean_raw_lyrics_for_training(raw_lyrics) or raw_lyrics
        return LyricsSelection(fallback_lyrics, "", rejection_reason)

    return LyricsSelection(formatted_lyrics, formatted_lyrics)


def get_formatted_lyrics_rejection_reason(raw_lyrics: str, formatted_lyrics: str) -> str:
    """Return why formatted lyrics should be rejected, or an empty string."""

    if _has_excessive_line_repetition(formatted_lyrics):
        return "repetitive formatted lyrics"
    if _has_low_raw_overlap(raw_lyrics, formatted_lyrics):
        return "low overlap with raw lyrics"
    return ""


def _has_excessive_line_repetition(lyrics: str) -> bool:
    if _has_excessive_word_repetition(lyrics):
        return True

    lines = _content_lines(lyrics)
    if len(lines) < _MIN_REPETITION_LINES:
        return False

    line_counts = Counter(lines)
    top_line_share = max(line_counts.values()) / len(lines)
    unique_line_share = len(line_counts) / len(lines)

    return (
        top_line_share >= _MAX_TOP_LINE_SHARE
        or unique_line_share <= _MIN_UNIQUE_LINE_SHARE
    )


def _has_excessive_word_repetition(lyrics: str) -> bool:
    words = [
        word.strip("'").lower()
        for word in _WORD_RE.findall(str(lyrics or ""))
        if len(word.strip("'")) > 1
    ]
    if len(words) < _MIN_WORDS_FOR_REPETITION:
        return False

    counts = Counter(words)
    if max(counts.values()) / len(words) >= _MAX_TOP_WORD_SHARE:
        return True

    current_word = ""
    current_run = 0
    for word in words:
        if word == current_word:
            current_run += 1
        else:
            current_word = word
            current_run = 1
        if current_run >= _MAX_WORD_RUN:
            return True
    return False


def _has_low_raw_overlap(raw_lyrics: str, formatted_lyrics: str) -> bool:
    raw_words = _content_words(raw_lyrics)
    formatted_words = _content_words(formatted_lyrics)

    if (
        len(raw_words) < _MIN_RAW_WORDS_FOR_OVERLAP
        or len(formatted_words) < _MIN_FORMATTED_WORDS_FOR_OVERLAP
    ):
        return False

    overlap = len(raw_words.intersection(formatted_words)) / len(formatted_words)
    return overlap < _MIN_FORMATTED_RAW_OVERLAP


def _content_lines(lyrics: str) -> list[str]:
    lines = []
    for line in str(lyrics or "").splitlines():
        stripped = line.strip()
        if not stripped or _SECTION_TAG_RE.match(stripped):
            continue
        normalized = " ".join(_WORD_RE.findall(stripped.lower()))
        if normalized:
            lines.append(normalized)
    return lines


def _content_words(lyrics: str) -> set[str]:
    words = {
        word.strip("'")
        for word in _WORD_RE.findall(str(lyrics or "").lower())
        if len(word.strip("'")) > 2
    }
    return {
        word
        for word in words
        if word not in _STOP_WORDS and not word.startswith("verse")
    }
