"""Parse plain and sectioned lyric sidecar text files."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


_MARKDOWN_HEADER_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?P<label>[^:#\n]+?)\s*(?::\s*(?P<inline>.*))?\s*$"
)
_LABEL_HEADER_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z][A-Za-z _-]{1,40})\s*:\s*(?P<inline>.*)$"
)
_METADATA_ENTRY_RE = re.compile(r"^\s*[-*]?\s*(?P<key>[A-Za-z][\w -]*)\s*:\s*(?P<value>.*?)\s*$")
_SECTION_BY_LABEL = {
    "caption": "caption",
    "description": "caption",
    "music description": "caption",
    "prompt": "caption",
    "lyric": "lyrics",
    "lyrics": "lyrics",
    "song lyric": "lyrics",
    "song lyrics": "lyrics",
    "meta": "metadata",
    "metadata": "metadata",
    "music metadata": "metadata",
    "comment": "notes",
    "comments": "notes",
    "notes": "notes",
    "revision": "notes",
    "revision note": "notes",
    "revision notes": "notes",
}
_METADATA_KEYS = {
    "bpm": "bpm",
    "duration": "duration",
    "genre": "genre",
    "genres": "genres",
    "instrumental": "instrumental",
    "is instrumental": "is_instrumental",
    "is_instrumental": "is_instrumental",
    "key": "keyscale",
    "keyscale": "keyscale",
    "language": "language",
    "time signature": "timesignature",
    "time_signature": "timesignature",
    "timesig": "timesignature",
    "timesignature": "timesignature",
    "vocal language": "vocal_language",
    "vocal_language": "vocal_language",
}


@dataclass(frozen=True)
class ParsedLyricsFile:
    """Structured content extracted from a lyric sidecar file."""

    lyrics: str = ""
    caption: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_lyrics_text_file(text: str) -> ParsedLyricsFile:
    """Parse plain lyrics or a ``# Caption/# Lyrics/# Metadata`` text file."""

    cleaned_text = _strip_outer_fence(str(text or ""))
    lines = cleaned_text.splitlines()
    sections: dict[str, list[str]] = {}
    current_section = ""
    saw_section = False

    for line in lines:
        section, inline = _match_section_header(line)
        if section:
            current_section = section
            saw_section = True
            sections.setdefault(current_section, [])
            if inline:
                sections[current_section].append(inline)
            continue

        if current_section:
            sections.setdefault(current_section, []).append(line)

    if not saw_section:
        return ParsedLyricsFile(lyrics=cleaned_text.strip())

    lyrics = _join_section_lines(sections.get("lyrics", []))
    caption = _caption_text(sections.get("caption", []))
    metadata = _parse_metadata(sections.get("metadata", []))
    return ParsedLyricsFile(lyrics=lyrics, caption=caption, metadata=metadata)


def _strip_outer_fence(text: str) -> str:
    """Remove one wrapping Markdown code fence when present."""

    lines = str(text or "").strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines).strip()


def _match_section_header(line: str) -> tuple[str, str]:
    """Return a normalized section name and inline text for a header line."""

    match = _MARKDOWN_HEADER_RE.match(line) or _LABEL_HEADER_RE.match(line)
    if not match:
        return "", ""

    label = _normalize_label(match.group("label"))
    section = _SECTION_BY_LABEL.get(label, "")
    if not section:
        return "", ""
    return section, (match.group("inline") or "").strip()


def _normalize_label(label: str) -> str:
    """Normalize free-form section or metadata labels."""

    normalized = re.sub(r"[\s_-]+", " ", str(label or "").strip().lower())
    return normalized.strip(" :")


def _join_section_lines(lines: list[str]) -> str:
    """Join section lines while trimming only outer blank lines."""

    trimmed = [line.rstrip() for line in lines]
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed).strip()


def _caption_text(lines: list[str]) -> str:
    """Collapse caption section lines into one caption string."""

    return " ".join(line.strip() for line in lines if line.strip())


def _parse_metadata(lines: list[str]) -> dict[str, Any]:
    """Parse recognized ``key: value`` entries from a metadata section."""

    metadata: dict[str, Any] = {}
    for line in lines:
        match = _METADATA_ENTRY_RE.match(line)
        if not match:
            continue
        key = _METADATA_KEYS.get(_normalize_label(match.group("key")))
        value = match.group("value").strip()
        if key and value:
            metadata[key] = _parse_metadata_value(key, value)
    return metadata


def _parse_metadata_value(key: str, value: str) -> Any:
    """Convert simple boolean and numeric metadata values."""

    normalized = value.strip().strip("\"'")
    lower = normalized.lower()
    if key in {"instrumental", "is_instrumental"}:
        if lower in {"true", "yes", "1", "y"}:
            return True
        if lower in {"false", "no", "0", "n"}:
            return False
    if key in {"bpm", "duration"}:
        try:
            return int(float(normalized))
        except ValueError:
            return normalized
    return normalized
