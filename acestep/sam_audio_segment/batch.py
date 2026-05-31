"""Batch folder processing for SAM-Audio segmentation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from loguru import logger

from acestep.audio_processing.json_io import write_json
from acestep.audio_processing.media_io import is_supported_media

from .paths import create_run_dir, safe_media_stem
from .progress import ProgressCallback, report_progress
from .service import SamAudioService
from .settings import SamAudioSettings


def iter_media_files(input_folder: str | Path, recursive: bool = False) -> list[Path]:
    """Return supported audio/video files in a folder."""

    root = Path(input_folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and is_supported_media(path))


def run_batch_sam_audio(
    input_folder: str,
    output_folder: str,
    recursive: bool,
    settings: SamAudioSettings,
    progress_callback: ProgressCallback | None = None,
) -> Iterator[tuple[str, list[str]]]:
    """Process every supported media file in ``input_folder``."""

    status_lines: list[str] = []
    generated_files: list[str] = []
    rows: list[dict[str, object]] = []
    try:
        files = iter_media_files(input_folder, recursive=recursive)
        run_dir = create_run_dir(output_folder.strip() or None)
    except ValueError as exc:
        yield str(exc), []
        return

    status_lines.extend([f"Found {len(files)} media file(s).", f"Saving outputs under: {run_dir}"])
    report_progress(progress_callback, 0.0, f"Found {len(files)} media file(s)")
    yield _render_status(status_lines), generated_files
    service = SamAudioService(settings)
    try:
        for index, source in enumerate(files, start=1):
            started_at = time.time()
            status_lines.append(f"[{index}/{len(files)}] Processing {source.name}")
            report_progress(
                progress_callback,
                (index - 1) / max(1, len(files)),
                f"[{index}/{len(files)}] Processing {source.name}",
            )
            service.progress_callback = _file_progress_callback(
                progress_callback,
                index,
                len(files),
            )
            yield _render_status(status_lines), generated_files
            try:
                result = service.process_file(source, run_dir, output_stem=safe_media_stem(source))
                generated_files.extend(result.file_list())
                rows.append(
                    {
                        "source_path": str(source),
                        "status": "completed",
                        "duration_seconds": round(max(0.0, time.time() - started_at), 3),
                        "outputs": result.file_list(),
                    }
                )
                status_lines.append(f"[{index}/{len(files)}] Done: {source.name}")
            except Exception as exc:
                logger.exception("[sam_audio_batch] Failed for {}", source)
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
            if manifest_path not in generated_files:
                generated_files.append(manifest_path)
            yield _render_status(status_lines), generated_files
    finally:
        service.unload()

    completed = sum(1 for row in rows if row["status"] == "completed")
    status_lines.append(f"Batch complete: {completed}/{len(files)} item(s) processed.")
    report_progress(progress_callback, 1.0, f"Batch complete: {completed}/{len(files)}")
    yield _render_status(status_lines), generated_files


def _file_progress_callback(
    callback: ProgressCallback | None,
    index: int,
    total: int,
) -> ProgressCallback | None:
    """Return a callback that maps one file's progress into batch progress."""

    if callback is None:
        return None

    def _report(file_fraction: float, message: str) -> None:
        overall = ((index - 1) + file_fraction) / max(1, total)
        report_progress(callback, overall, f"[{index}/{total}] {message}")

    return _report


def _write_manifest(
    run_dir: Path,
    rows: list[dict[str, object]],
    settings: SamAudioSettings,
) -> str:
    """Persist batch metadata."""

    return write_json(
        run_dir / "sam_audio_batch_manifest.json",
        {
            "_meta": {
                "format": "ace_step_sam_audio_batch",
                "version": 1,
                "updated_at_unix": time.time(),
            },
            "settings": settings.to_payload(),
            "items": rows,
        },
    )


def _render_status(lines: list[str]) -> str:
    """Return recent status lines for Gradio."""

    return "\n".join(lines[-80:])
