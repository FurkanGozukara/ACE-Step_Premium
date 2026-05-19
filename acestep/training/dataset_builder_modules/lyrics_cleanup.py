"""Conservative cleanup for rejected LM-formatted lyrics."""

from __future__ import annotations

import re


_SECTION_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")
_NOISE_LINES = {"transcript", "lyrics", "subtitles", "in the city city of"}
_NOISE_SUBSTRINGS = {"news out of", "transcript"}
_MAX_LINE_WORDS = 12
_MAX_LINES = 48
_MIN_CONTENT_WORDS = 3
_MAX_REPEATED_WORDS = 4
_MAX_REPEATED_PHRASES = 2


def clean_raw_lyrics_for_training(raw_lyrics: str) -> str:
    """Return a structured, lower-noise fallback for training lyrics.

    Args:
        raw_lyrics: Original lyric text or transcript text.

    Returns:
        Cleaned lyrics with section tags. Returns an empty string when no usable
        lyric lines remain.
    """

    lines = _clean_content_lines(raw_lyrics)
    if not lines:
        return ""
    if _has_section_tags(raw_lyrics):
        return "\n".join(lines)
    return _add_basic_sections(lines)


def _clean_content_lines(raw_lyrics: str) -> list[str]:
    cleaned: list[str] = []
    for line in str(raw_lyrics or "").splitlines():
        line = _normalize_line(line)
        if not line:
            continue
        if _SECTION_TAG_RE.match(line):
            cleaned.append(line)
            continue
        if _is_noise_line(line):
            continue
        for chunk in _split_long_line(line):
            chunk = _collapse_repetition(chunk)
            if chunk and not _is_noise_line(chunk):
                cleaned.append(chunk)
                if len([item for item in cleaned if not _SECTION_TAG_RE.match(item)]) >= _MAX_LINES:
                    return cleaned
    return cleaned


def _normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    return line.strip(" -_\t")


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    if lower in _NOISE_LINES:
        return True
    if any(noise in lower for noise in _NOISE_SUBSTRINGS):
        return True
    return len(re.findall(r"\w+", lower)) < _MIN_CONTENT_WORDS


def _split_long_line(line: str) -> list[str]:
    words = line.split()
    if len(words) <= _MAX_LINE_WORDS:
        return [line]
    return [
        " ".join(words[index : index + _MAX_LINE_WORDS])
        for index in range(0, len(words), _MAX_LINE_WORDS)
    ]


def _collapse_repetition(line: str) -> str:
    tokens = line.split()
    if not tokens:
        return ""

    tokens = _collapse_repeated_words(tokens)
    for phrase_len in range(4, 1, -1):
        tokens = _collapse_repeated_phrases(tokens, phrase_len)
    return " ".join(tokens).strip()


def _collapse_repeated_words(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    previous = ""
    repeat_count = 0
    for token in tokens:
        normalized = token.strip(".,!?;:\"'()[]{}").lower()
        if normalized and normalized == previous:
            repeat_count += 1
        else:
            previous = normalized
            repeat_count = 1
        if repeat_count <= _MAX_REPEATED_WORDS:
            collapsed.append(token)
    return collapsed


def _collapse_repeated_phrases(tokens: list[str], phrase_len: int) -> list[str]:
    collapsed: list[str] = []
    index = 0
    while index < len(tokens):
        phrase = _phrase_key(tokens[index : index + phrase_len])
        if len(phrase) < phrase_len:
            collapsed.extend(tokens[index:])
            break

        repeat_count = 1
        next_index = index + phrase_len
        while _phrase_key(tokens[next_index : next_index + phrase_len]) == phrase:
            repeat_count += 1
            next_index += phrase_len

        kept_repeats = min(repeat_count, _MAX_REPEATED_PHRASES)
        collapsed.extend(tokens[index : index + phrase_len * kept_repeats])
        index = next_index
    return collapsed


def _phrase_key(tokens: list[str]) -> tuple[str, ...]:
    return tuple(token.strip(".,!?;:\"'()[]{}").lower() for token in tokens if token.strip())


def _has_section_tags(lyrics: str) -> bool:
    return any(_SECTION_TAG_RE.match(line.strip()) for line in str(lyrics or "").splitlines())


def _add_basic_sections(lines: list[str]) -> str:
    content = [line for line in lines if not _SECTION_TAG_RE.match(line)]
    if not content:
        return ""

    sections: list[str] = []
    intro = content[:2]
    body = content[2:]
    if intro:
        sections.extend(["[Intro]", *intro, ""])
    for verse_index, start in enumerate(range(0, len(body), 8), start=1):
        sections.extend([f"[Verse {verse_index}]", *body[start : start + 8], ""])
    return "\n".join(sections).strip()
