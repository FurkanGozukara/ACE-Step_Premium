"""Record selection and table helpers for the generated-song library."""

from __future__ import annotations

from typing import Any


def date_choices_for_records(records: list[dict[str, Any]]) -> list[str]:
    """Return newest-first unique date labels from display records.

    Args:
        records: Generated-song display records.

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
    return [record for record in records if record.get("created_day") == day]


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
