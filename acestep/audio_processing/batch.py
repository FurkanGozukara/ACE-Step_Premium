"""Batch folder processing for ACE-Step audio processing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from loguru import logger

from .auto_editor_workflow import export_auto_editor_workflow, workflow_export_enabled
from .file_processor import process_media_file
from .json_io import write_json
from .media_io import is_supported_media
from .runs import create_audio_processing_run_dir, safe_media_stem
from .settings import AudioProcessingSettings


def iter_media_files(input_folder: str | Path, recursive: bool = False) -> list[Path]:
    """Return supported media files in a folder."""

    root = Path(input_folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and is_supported_media(path))


def run_batch_audio_processing(
    input_folder: str,
    output_folder: str,
    recursive: bool,
    settings: AudioProcessingSettings,
) -> Iterator[tuple[str, list[str]]]:
    """Process every supported audio/video file in a folder.

    Args:
        input_folder: Folder containing media files.
        output_folder: Optional destination root. Blank values use ACE-Step outputs.
        recursive: Whether to scan subfolders.
        settings: Audio-processing settings.

    Yields:
        Tuple of status text and generated file paths.
    """

    status_lines: list[str] = []
    generated_files: list[str] = []
    rows: list[dict[str, object]] = []
    try:
        files = iter_media_files(input_folder, recursive=recursive)
        run_dir = create_audio_processing_run_dir(output_folder.strip() or None)
    except ValueError as exc:
        yield str(exc), []
        return

    status_lines.append(f"Found {len(files)} media file(s).")
    output_label = (
        "Saving workflow exports under"
        if workflow_export_enabled(settings.workflow_export)
        else "Saving processed audio under"
        if settings.export_audio_only
        else "Saving processed outputs under"
    )
    status_lines.append(f"{output_label}: {run_dir}")
    yield _render_status(status_lines), generated_files

    for index, source in enumerate(files, start=1):
        started_at = time.time()
        action = (
            "Exporting workflow"
            if workflow_export_enabled(settings.workflow_export)
            else "Processing"
        )
        status_lines.append(f"[{index}/{len(files)}] {action} {source.name}")
        yield _render_status(status_lines), generated_files
        try:
            result = None
            if workflow_export_enabled(settings.workflow_export):
                outputs = [
                    export_auto_editor_workflow(
                        source,
                        run_dir,
                        safe_media_stem(source),
                        settings.workflow_export,
                        settings.trim_settings(),
                    )
                ]
            else:
                result = process_media_file(
                    source,
                    run_dir,
                    settings,
                    output_stem=safe_media_stem(source),
                )
                outputs = result.file_list()
            rows.append(
                {
                    "source_path": str(source),
                    "status": "completed",
                    "duration_seconds": round(max(0.0, time.time() - started_at), 3),
                    "outputs": outputs,
                    **_batch_metrics_row(result),
                }
            )
            generated_files.extend(outputs)
            status_lines.append(f"[{index}/{len(files)}] Done: {source.name}")
        except Exception as exc:
            logger.exception("[audio_processing_batch] Failed for {}", source)
            rows.append(
                {
                    "source_path": str(source),
                    "status": "failed",
                    "message": str(exc),
                    "duration_seconds": round(max(0.0, time.time() - started_at), 3),
                }
            )
            status_lines.append(f"[{index}/{len(files)}] Failed: {exc}")
        manifest_path = _write_manifest(run_dir, rows, settings)
        generated_files.append(manifest_path)
        status_lines.append(f"Manifest: {manifest_path}")
        yield _render_status(status_lines), generated_files

    completed = sum(1 for row in rows if row["status"] == "completed")
    status_lines.append(f"Batch complete: {completed}/{len(files)} item(s) processed.")
    yield _render_status(status_lines), generated_files


def _write_manifest(
    run_dir: Path,
    rows: list[dict[str, object]],
    settings: AudioProcessingSettings,
) -> str:
    """Persist batch processing metadata."""

    return write_json(
        run_dir / "audio_processing_batch_manifest.json",
        {
            "_meta": {
                "format": "ace_step_audio_processing_batch",
                "version": 1,
                "updated_at_unix": time.time(),
            },
            "settings": settings.to_payload(),
            "items": rows,
        },
    )


def _batch_metrics_row(result: object) -> dict[str, object]:
    """Return audio metrics when media processing produced them."""

    processed_audio = getattr(result, "processed_audio", None)
    if processed_audio is None:
        return {}
    return {
        "lufs_before": processed_audio.lufs_before,
        "lufs_after": processed_audio.lufs_after,
    }


def _render_status(lines: list[str]) -> str:
    """Return recent status lines for the Gradio textbox."""

    return "\n".join(lines[-80:])
