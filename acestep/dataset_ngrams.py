"""Word n-gram extraction for the Dataset page browser."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from acestep.dataset_preview import sample_title
from acestep.training.dataset_builder_modules.models import AudioSample

NGRAM_LIMIT = 25
NGRAM_SIZES = (1, 2, 3, 4, 5, 6)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def build_ngram_tables(
    samples: Iterable[AudioSample] | None,
    limit: int = NGRAM_LIMIT,
) -> tuple[list[list[Any]], ...]:
    """Return top 1-gram through 6-gram tables for captions and style text."""

    sample_list = list(samples or [])
    occurrences, song_indices = _collect_ngram_stats(sample_list)
    return tuple(
        _table_for_size(occurrences[size], song_indices[size], limit)
        for size in NGRAM_SIZES
    )


def selected_ngram_from_event(
    samples: Iterable[AudioSample] | None,
    gram_size: int,
    event: Any,
    limit: int = NGRAM_LIMIT,
) -> str:
    """Return the n-gram selected from a top-n table event."""

    row_index = selected_row_index(event)
    if row_index is None:
        return ""

    tables = build_ngram_tables(samples, limit)
    size_index = int(gram_size) - 1
    if size_index < 0 or size_index >= len(tables):
        return ""

    table = tables[size_index]
    if row_index < 0 or row_index >= len(table):
        return ""
    return str(table[row_index][0])


def song_rows_for_ngram(
    samples: Iterable[AudioSample] | None,
    gram_size: int,
    gram: str,
) -> list[list[Any]]:
    """Return song rows that contain the selected word n-gram."""

    sample_list = list(samples or [])
    normalized = _normalize_selected_gram(gram, gram_size)
    if not normalized:
        return []

    rows: list[list[Any]] = []
    for index, sample in enumerate(sample_list):
        count = _count_ngram(_tokens_for_sample(sample), normalized)
        if count:
            rows.append(
                [
                    index,
                    sample_title(sample),
                    count,
                    _format_duration(sample.duration),
                    sample.genre or "-",
                ]
            )
    return rows


def selected_song_index_from_event(
    samples: Iterable[AudioSample] | None,
    gram_size: int,
    gram: str,
    event: Any,
) -> int | None:
    """Return the dataset sample index selected from a gram song table."""

    row_index = selected_row_index(event)
    if row_index is None:
        return None
    rows = song_rows_for_ngram(samples, gram_size, gram)
    if row_index < 0 or row_index >= len(rows):
        return None
    return int(rows[row_index][0])


def selected_row_index(event: Any) -> int | None:
    """Return the row index from a Gradio table selection event."""

    index = getattr(event, "index", None)
    if isinstance(index, (list, tuple)) and index:
        index = index[0]
    try:
        return int(index)
    except (TypeError, ValueError):
        return None
def _collect_ngram_stats(
    samples: list[AudioSample],
) -> tuple[dict[int, Counter[str]], dict[int, defaultdict[str, set[int]]]]:
    """Count n-gram hits and containing songs by gram size."""

    occurrences = {size: Counter() for size in NGRAM_SIZES}
    song_indices = {size: defaultdict(set) for size in NGRAM_SIZES}

    for sample_index, sample in enumerate(samples):
        tokens = _tokens_for_sample(sample)
        for size in NGRAM_SIZES:
            sample_counts = Counter(_ngrams(tokens, size))
            for gram, count in sample_counts.items():
                occurrences[size][gram] += count
                song_indices[size][gram].add(sample_index)
    return occurrences, song_indices
def _table_for_size(
    occurrences: Counter[str],
    song_indices: defaultdict[str, set[int]],
    limit: int,
) -> list[list[Any]]:
    """Format one n-gram size as browser rows."""

    grams = sorted(
        occurrences,
        key=lambda gram: (-len(song_indices[gram]), -occurrences[gram], gram),
    )
    return [
        [gram, len(song_indices[gram]), occurrences[gram]]
        for gram in grams[: max(0, int(limit))]
    ]
def _tokens_for_sample(sample: AudioSample) -> list[str]:
    """Return normalized words from a sample's caption and style text."""

    return [
        match.group(0).lower()
        for match in _WORD_RE.finditer(_caption_style_text(sample))
    ]
def _caption_style_text(sample: AudioSample) -> str:
    """Return caption and style-like metadata for n-gram extraction."""

    values = (
        sample.caption,
        getattr(sample, "style", ""),
        sample.genre,
    )
    return " ".join(str(value or "").strip() for value in values if str(value or "").strip())
def _ngrams(tokens: list[str], size: int) -> list[str]:
    """Return joined token n-grams for one gram size."""

    if size <= 0 or len(tokens) < size:
        return []
    return [
        " ".join(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    ]
def _normalize_selected_gram(gram: str, gram_size: int) -> tuple[str, ...]:
    """Normalize a selected gram string to comparable tokens."""

    tokens = tuple(match.group(0).lower() for match in _WORD_RE.finditer(str(gram or "")))
    if len(tokens) != int(gram_size or 0):
        return ()
    return tokens
def _count_ngram(tokens: list[str], selected: tuple[str, ...]) -> int:
    """Count the selected n-gram in one token stream."""

    size = len(selected)
    if size <= 0 or len(tokens) < size:
        return 0
    return sum(
        1
        for index in range(len(tokens) - size + 1)
        if tuple(tokens[index : index + size]) == selected
    )
def _format_duration(duration: Any) -> str:
    """Format a duration value for the n-gram song table."""

    try:
        seconds = float(duration or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return f"{seconds:.1f}s" if seconds else "-"
