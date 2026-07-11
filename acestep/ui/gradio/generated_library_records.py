"""Record selection and table helpers for the generated-song library."""

from __future__ import annotations

from fnmatch import fnmatchcase
from math import ceil
from typing import Any


ALL_DAYS_CHOICE = "All Dates"
SEARCH_WILDCARDS = ("*", "?", "[")


def date_choices_for_records(
    records: list[dict[str, Any]],
    include_all: bool = False,
) -> list[str]:
    """Return newest-first unique date labels from display records.

    Args:
        records: Generated-song display records.
        include_all: Whether to prepend the all-days option.

    Returns:
        Unique non-empty ``created_day`` values in record order.
    """

    choices: list[str] = []
    seen: set[str] = set()
    for record in records:
        day = str(record.get("created_day") or "").strip()
        if day and day not in seen:
            choices.append(day)
            seen.add(day)
    if include_all and choices:
        return [ALL_DAYS_CHOICE, *choices]
    return choices


def filter_records_by_day(records: list[dict[str, Any]], selected_day: Any) -> list[dict[str, Any]]:
    """Return records matching the selected generated-song day.

    Args:
        records: Generated-song display records.
        selected_day: Date label chosen in the library filter.

    Returns:
        Records whose ``created_day`` matches ``selected_day``.
    """

    day = str(selected_day or "").strip()
    if not day:
        return []
    if day == ALL_DAYS_CHOICE:
        return list(records)
    return [record for record in records if record.get("created_day") == day]


def filter_records_by_search(
    records: list[dict[str, Any]],
    search_query: Any,
) -> list[dict[str, Any]]:
    """Return records matching a plain-text or wildcard search query."""

    query = _normalize_search_value(search_query)
    if not query:
        return list(records)
    return [record for record in records if _record_matches_search(record, query)]


def paginate_records(
    records: list[dict[str, Any]],
    page: Any,
    page_size: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Return one page of records plus normalized pagination metadata."""

    total_count = len(records)
    if total_count <= 0 or page_size <= 0:
        return [], 1, 0, total_count

    total_pages = max(1, ceil(total_count / page_size))
    current_page = _coerce_page_number(page)
    current_page = min(max(1, current_page), total_pages)
    start = (current_page - 1) * page_size
    return records[start : start + page_size], current_page, total_pages, total_count


def find_record(item_id: str | None, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the selected record or the first record."""

    if not records:
        return None
    for record in records:
        if record.get("id") == item_id:
            return record
    return records[0]


def record_from_table_event(
    records: list[dict[str, Any]],
    event: Any,
) -> dict[str, Any] | None:
    """Return the filtered record corresponding to a Dataframe selection.

    Args:
        records: Records currently displayed in the filtered table.
        event: Gradio ``SelectData``-like object with an ``index`` attribute.

    Returns:
        The clicked row's record, or ``None`` when the selection is not a row.
    """

    row_index = _row_index_from_event(event)
    if row_index is None or row_index < 0 or row_index >= len(records):
        return None
    return records[row_index]


def records_to_table(records: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert records to a Dataframe-compatible table."""

    return [
        [
            record.get("created_display") or record.get("created", "")[:19],
            record.get("title", ""),
            record.get("duration", ""),
            record.get("model", ""),
            record.get("format", ""),
            record.get("score", ""),
        ]
        for record in records
    ]


def details_markdown(record: dict[str, Any]) -> str:
    """Format one generated-song detail card as Markdown."""

    return "\n".join(
        [
            f"### {record.get('title', 'Generated song')}",
            f"- Created: `{record.get('created_display') or record.get('created', '')}`",
            f"- Model: `{record.get('model', '')}`",
            f"- Duration: `{record.get('duration', '')}`",
            f"- Format: `{record.get('format', '')}`",
            f"- Score: `{record.get('score', '')}`",
            f"- Audio: `{record.get('audio_path') or 'missing'}`",
            f"- Metadata: `{record.get('metadata_path', '')}`",
            f"- Manifest: `{record.get('manifest_path', '')}`",
            "",
            record.get("caption", ""),
        ]
    )


def _row_index_from_event(event: Any) -> int | None:
    """Extract a table row index from a Gradio selection event."""

    index = getattr(event, "index", None)
    if isinstance(index, int):
        return index
    if isinstance(index, (list, tuple)) and index and isinstance(index[0], int):
        return index[0]
    return None


def _coerce_page_number(value: Any) -> int:
    """Return a positive 1-based page number."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _record_matches_search(record: dict[str, Any], query: str) -> bool:
    """Return whether a record matches the normalized search query."""

    values = [_normalize_search_value(value) for value in _searchable_record_values(record)]
    haystack = "\n".join(value for value in values if value)
    if _contains_wildcard(query):
        return any(_wildcard_matches(value, query) for value in [haystack, *values] if value)
    return query in haystack


def _searchable_record_values(record: dict[str, Any]) -> list[Any]:
    """Return compact fields users naturally search in the library."""

    return [
        record.get("title"),
        record.get("caption"),
        record.get("lyrics"),
        record.get("model"),
        record.get("format"),
        record.get("score"),
        record.get("created"),
        record.get("created_display"),
        record.get("created_day"),
        record.get("audio_path"),
        record.get("metadata_path"),
        record.get("manifest_path"),
    ]


def _normalize_search_value(value: Any) -> str:
    """Return a case-insensitive, single-line search string."""

    return " ".join(str(value or "").casefold().split())


def _contains_wildcard(query: str) -> bool:
    """Return whether the query uses shell-style wildcard syntax."""

    return any(marker in query for marker in SEARCH_WILDCARDS)


def _wildcard_matches(value: str, query: str) -> bool:
    """Return whether a normalized value matches a wildcard query."""

    contained_query = query if query.startswith("*") and query.endswith("*") else f"*{query}*"
    return fnmatchcase(value, query) or fnmatchcase(value, contained_query)
