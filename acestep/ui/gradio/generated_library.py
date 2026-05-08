"""Generated-song library helpers for Gradio pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.ui.gradio.events.results.output_manager import get_results_dir


TABLE_HEADERS = ["Created", "Title", "Duration", "Model", "Format", "Score"]


def refresh_library() -> tuple[Any, ...]:
    """Return UI updates for the generated-song library tab."""

    records = scan_generated_songs()
    choices = [(record["label"], record["id"]) for record in records]
    selected = records[0]["id"] if records else None
    table = _records_to_table(records)
    audio, details, lyrics, metadata = select_library_item(selected, records)
    return (
        records,
        gr.update(choices=choices, value=selected),
        table,
        audio,
        details,
        lyrics,
        metadata,
    )


def select_library_item(item_id: str | None, records: list[dict[str, Any]] | None) -> tuple[Any, ...]:
    """Return detail-panel updates for one generated library item."""

    record = _find_record(item_id, records or [])
    if record is None:
        return None, "No generated song selected.", "", {}
    return (
        record.get("audio_path"),
        _details_markdown(record),
        record.get("lyrics", ""),
        record.get("metadata", {}),
    )


def scan_generated_songs(results_dir: str | Path | None = None) -> list[dict[str, Any]]:
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
                records.append(_record_from_sample(manifest_path, manifest, sample))
    return records


def _record_from_sample(
    manifest_path: Path,
    manifest: dict[str, Any],
    sample: dict[str, Any],
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
    audio_path = str(sample.get("audio_path") or "")
    return {
        "id": f"{manifest_path}:{sample.get('sample_index', 0)}",
        "label": f"{created[:19]} | {title}",
        "created": created,
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


def _records_to_table(records: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert records to a Dataframe-compatible table."""

    return [
        [
            record.get("created", "")[:19],
            record.get("title", ""),
            record.get("duration", ""),
            record.get("model", ""),
            record.get("format", ""),
            record.get("score", ""),
        ]
        for record in records
    ]


def _details_markdown(record: dict[str, Any]) -> str:
    """Format one generated-song detail card as Markdown."""

    return "\n".join(
        [
            f"### {record.get('title', 'Generated song')}",
            f"- Created: `{record.get('created', '')}`",
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


def _find_record(item_id: str | None, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the selected record or the first record."""

    if not records:
        return None
    for record in records:
        if record.get("id") == item_id:
            return record
    return records[0]


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
