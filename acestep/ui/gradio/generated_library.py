"""Generated-song library helpers for Gradio pages."""

from __future__ import annotations

import json
from datetime import tzinfo
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.ui.gradio.events.results.output_manager import get_results_dir
from acestep.ui.gradio.generated_library_records import (
    date_choices_for_records,
    details_markdown as _details_markdown,
    filter_records_by_day,
    filter_records_by_search,
    find_record as _find_record,
    paginate_records,
    record_from_table_event,
    records_to_table as _records_to_table,
)
from acestep.ui.gradio.generated_library_time import (
    format_library_created_date,
    format_library_created_time,
    resolve_library_timezone,
)


TABLE_HEADERS = ["Created", "Title", "Duration", "Model", "Format", "Score"]
LIBRARY_PAGE_SIZE = 50


def refresh_library(
    browser_timezone: str | None = None,
    search_query: Any = "",
) -> tuple[Any, ...]:
    """Return UI updates for the generated-song library tab."""

    local_timezone = resolve_library_timezone(browser_timezone)
    records = scan_generated_songs(local_timezone=local_timezone)
    choices = date_choices_for_records(records, include_all=True)
    selected_day = choices[0] if choices else None
    view = _library_page_outputs(records, selected_day, search_query, page=1)
    return (
        records,
        *view[:2],
        gr.update(choices=choices, value=selected_day),
        *view[2:],
    )


def filter_library_by_date(
    selected_day: str | None,
    records: list[dict[str, Any]] | None,
    search_query: Any = "",
) -> tuple[Any, ...]:
    """Return table and detail updates for the selected generated-song date."""

    filtered_records = filter_records_by_day(records or [], selected_day)
    filtered_records = filter_records_by_search(filtered_records, search_query)
    selected = filtered_records[0]["id"] if filtered_records else None
    audio, details, lyrics, metadata = select_library_item(selected, filtered_records)
    return (
        filtered_records,
        _records_to_table(filtered_records),
        audio,
        details,
        lyrics,
        metadata,
    )


def filter_library_view(
    selected_day: str | None,
    search_query: Any,
    records: list[dict[str, Any]] | None,
) -> tuple[Any, ...]:
    """Return first-page library updates for the selected date and search query."""

    return _library_page_outputs(records or [], selected_day, search_query, page=1)


def previous_library_page(
    selected_day: str | None,
    search_query: Any,
    records: list[dict[str, Any]] | None,
    current_page: Any,
) -> tuple[Any, ...]:
    """Return library updates for the previous page."""

    return _library_page_outputs(
        records or [],
        selected_day,
        search_query,
        page=_coerce_page(current_page) - 1,
    )


def next_library_page(
    selected_day: str | None,
    search_query: Any,
    records: list[dict[str, Any]] | None,
    current_page: Any,
) -> tuple[Any, ...]:
    """Return library updates for the next page."""

    return _library_page_outputs(
        records or [],
        selected_day,
        search_query,
        page=_coerce_page(current_page) + 1,
    )


def select_library_item(
    item_id: str | None, records: list[dict[str, Any]] | None
) -> tuple[Any, ...]:
    """Return detail-panel updates for one generated library item."""

    record = _find_record(item_id, records or [])
    if record is None:
        return None, "No generated song selected.", "", ""
    return (
        record.get("audio_path"),
        _details_markdown(record),
        record.get("lyrics", ""),
        _metadata_display_json(record.get("metadata", {})),
    )


def select_library_table_item(
    records: list[dict[str, Any]] | None,
    evt: gr.SelectData,
) -> tuple[Any, ...]:
    """Return detail-panel updates for the clicked generated-song table row."""

    record = record_from_table_event(records or [], evt)
    if record is None:
        return None, "No generated song selected.", "", ""
    return (
        record.get("audio_path"),
        _details_markdown(record),
        record.get("lyrics", ""),
        _metadata_display_json(record.get("metadata", {})),
    )


def _metadata_display_json(metadata: Any) -> str:
    """Return metadata as copy-friendly JSON text without a heavy DOM tree."""

    if not metadata:
        return ""
    try:
        return json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps({"metadata": str(metadata)}, ensure_ascii=False, indent=2)


def _library_page_outputs(
    records: list[dict[str, Any]],
    selected_day: Any,
    search_query: Any,
    page: Any,
) -> tuple[Any, ...]:
    """Return current-page records, controls, table, and selected detail outputs."""

    filtered_records = filter_records_by_day(records, selected_day)
    filtered_records = filter_records_by_search(filtered_records, search_query)
    page_records, current_page, total_pages, total_count = paginate_records(
        filtered_records,
        page,
        LIBRARY_PAGE_SIZE,
    )
    selected = page_records[0]["id"] if page_records else None
    audio, details, lyrics, metadata = select_library_item(selected, page_records)
    return (
        page_records,
        current_page,
        _records_to_table(page_records),
        _library_page_status(
            page_records=page_records,
            current_page=current_page,
            total_pages=total_pages,
            total_count=total_count,
        ),
        gr.update(interactive=current_page > 1),
        gr.update(interactive=total_pages > 0 and current_page < total_pages),
        audio,
        details,
        lyrics,
        metadata,
    )


def _library_page_status(
    page_records: list[dict[str, Any]],
    current_page: int,
    total_pages: int,
    total_count: int,
) -> str:
    """Return a compact pagination status label."""

    if total_count <= 0 or total_pages <= 0:
        return "No songs match the current filters."
    start = ((current_page - 1) * LIBRARY_PAGE_SIZE) + 1
    end = start + len(page_records) - 1
    return (
        f"Showing {start}-{end} of {total_count} songs "
        f"(50 per page). Page {current_page} of {total_pages}."
    )


def _coerce_page(value: Any) -> int:
    """Return a positive page number."""

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def scan_generated_songs(
    results_dir: str | Path | None = None, local_timezone: tzinfo | None = None
) -> list[dict[str, Any]]:
    """Scan generation manifests and sidecars into newest-first records."""

    root = Path(results_dir) if results_dir is not None else get_results_dir()
    manifests = sorted(root.glob("*/generation_manifest.json"), key=_mtime, reverse=True)
    records: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        samples = manifest.get("samples") if isinstance(manifest, dict) else None
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if isinstance(sample, dict):
                records.append(_record_from_sample(manifest_path, manifest, sample, local_timezone))
    return records


def _record_from_sample(
    manifest_path: Path,
    manifest: dict[str, Any],
    sample: dict[str, Any],
    local_timezone: tzinfo | None = None,
) -> dict[str, Any]:
    """Build a display record from a manifest sample entry."""

    request = manifest.get("request", {}) if isinstance(manifest.get("request"), dict) else {}
    params = sample.get("params", {}) if isinstance(sample.get("params"), dict) else {}
    metadata_value = str(sample.get("metadata_path") or "").strip()
    metadata_path = Path(metadata_value) if metadata_value else None
    sidecar = _read_json(metadata_path) if metadata_path and metadata_path.exists() else {}
    metadata = {**sidecar, "manifest": manifest, "sample": sample}
    title = _title_from_values(request.get("caption"), params.get("lyrics"), sample.get("key"))
    created = manifest.get("_meta", {}).get("finished_at_utc", "")
    created_display = format_library_created_time(created, local_timezone=local_timezone)
    created_day = format_library_created_date(created, local_timezone=local_timezone)
    audio_path = str(sample.get("audio_path") or "")
    return {
        "id": f"{manifest_path}:{sample.get('sample_index', 0)}",
        "created": created,
        "created_display": created_display,
        "created_day": created_day,
        "title": title,
        "duration": request.get("audio_duration") or params.get("duration") or "",
        "model": _model_label(request),
        "format": sample.get("audio_format") or manifest.get("audio_format") or "",
        "score": sample.get("score") or "",
        "audio_path": audio_path if Path(audio_path).exists() else None,
        "metadata_path": str(metadata_path) if metadata_path is not None else "",
        "manifest_path": str(manifest_path),
        "lyrics": request.get("lyrics") or params.get("lyrics") or "",
        "caption": request.get("caption") or params.get("prompt") or "",
        "metadata": metadata,
    }


def _title_from_values(caption: Any, lyrics: Any, fallback: Any) -> str:
    """Pick a compact title from caption, lyrics, or file key."""

    for value in (caption, lyrics, fallback):
        text = str(value or "").strip().replace("\n", " ")
        if text:
            return text[:90]
    return "Generated song"


def _model_label(request: dict[str, Any]) -> str:
    """Return a compact model/quantization label from request metadata."""

    runtime = request.get("runtime", {}) if isinstance(request.get("runtime"), dict) else {}
    dit_init = runtime.get("dit_last_init_params", {})
    dit_init = dit_init if isinstance(dit_init, dict) else {}
    model = request.get("config_path") or dit_init.get("config_path") or request.get("model") or ""
    quant = runtime.get("dit_quantization") or request.get("quantization") or ""
    return " / ".join(str(part) for part in (model, quant) if part) or "ACE-Step"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict on errors."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _mtime(path: Path) -> float:
    """Return modification time for sorting."""

    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
