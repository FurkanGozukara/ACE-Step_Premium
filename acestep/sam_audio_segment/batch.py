"""Batch folder processing for SAM-Audio segmentation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from loguru import logger

from acestep.audio_processing.json_io import write_json
from acestep.audio_processing.media_io import is_supported_media

from .batch_segment import batch_segment_prompts, settings_for_batch_segment_prompt
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
        prompts = batch_segment_prompts(settings)
        run_dir = create_run_dir(output_folder.strip() or None)
    except ValueError as exc:
        yield str(exc), []
        return

    total_units = len(files) * max(1, len(prompts))
    status_lines.extend([f"Found {len(files)} media file(s).", f"Saving outputs under: {run_dir}"])
    if prompts:
        status_lines.extend(
            [
                "Batch Segment prompts: " + ", ".join(prompt.text for prompt in prompts),
                f"Total SAM-Audio runs: {total_units}",
            ]
        )
    report_progress(progress_callback, 0.0, f"Found {len(files)} media file(s)")
    yield _render_status(status_lines), generated_files
    service = SamAudioService(settings)
    try:
        for index, source in enumerate(files, start=1):
            prompt_items = prompts or [None]
            for prompt_index, prompt in enumerate(prompt_items, start=1):
                started_at = time.time()
                unit_index = (index - 1) * len(prompt_items) + prompt_index
                label = _unit_label(source.name, prompt.text if prompt else None)
                status_prefix = _status_prefix(
                    index,
                    len(files),
                    prompt_index if prompts else None,
                    len(prompts) if prompts else None,
                )
                status_lines.append(f"{status_prefix} Processing {label}")
                report_progress(
                    progress_callback,
                    (unit_index - 1) / max(1, total_units),
                    f"{status_prefix} Processing {label}",
                )
                service.progress_callback = _unit_progress_callback(
                    progress_callback,
                    unit_index,
                    total_units,
                    label,
                )
                if prompt is not None:
                    service.settings = settings_for_batch_segment_prompt(settings, prompt)
                yield _render_status(status_lines), generated_files
                try:
                    output_stem = safe_media_stem(source)
                    if prompt is not None:
                        output_stem = f"{output_stem}_{prompt.suffix}"
                    result = service.process_file(source, run_dir, output_stem=output_stem)
                    generated_files.extend(result.file_list())
                    row: dict[str, object] = {
                        "source_path": str(source),
                        "status": "completed",
                        "duration_seconds": round(max(0.0, time.time() - started_at), 3),
                        "outputs": result.file_list(),
                    }
                    if prompt is not None:
                        row["batch_segment_prompt"] = prompt.text
                        row["batch_segment_suffix"] = prompt.suffix
                    rows.append(row)
                    status_lines.append(f"{status_prefix} Done: {label}")
                except Exception as exc:
                    logger.exception("[sam_audio_batch] Failed for {}", label)
                    row = {
                        "source_path": str(source),
                        "status": "failed",
                        "message": str(exc),
                        "duration_seconds": round(max(0.0, time.time() - started_at), 3),
                    }
                    if prompt is not None:
                        row["batch_segment_prompt"] = prompt.text
                        row["batch_segment_suffix"] = prompt.suffix
                    rows.append(row)
                    status_lines.append(f"{status_prefix} Failed: {exc}")
                manifest_path = _write_manifest(run_dir, rows, settings)
                if manifest_path not in generated_files:
                    generated_files.append(manifest_path)
                yield _render_status(status_lines), generated_files
    finally:
        service.unload()

    completed = sum(1 for row in rows if row["status"] == "completed")
    unit_label = "SAM-Audio run(s)" if prompts else "item(s)"
    status_lines.append(f"Batch complete: {completed}/{total_units} {unit_label} processed.")
    report_progress(progress_callback, 1.0, f"Batch complete: {completed}/{total_units}")
    yield _render_status(status_lines), generated_files


def _unit_progress_callback(
    callback: ProgressCallback | None,
    unit_index: int,
    total_units: int,
    label: str,
) -> ProgressCallback | None:
    """Return a callback that maps one SAM-Audio run into batch progress."""

    if callback is None:
        return None

    def _report(unit_fraction: float, message: str) -> None:
        overall = ((unit_index - 1) + unit_fraction) / max(1, total_units)
        report_progress(callback, overall, f"[{unit_index}/{total_units}] {label}: {message}")

    return _report


def _status_prefix(
    file_index: int,
    file_total: int,
    prompt_index: int | None,
    prompt_total: int | None,
) -> str:
    """Return a compact file/prompt status prefix."""

    if prompt_index is None or prompt_total is None:
        return f"[{file_index}/{file_total}]"
    return f"[{file_index}/{file_total}:{prompt_index}/{prompt_total}]"


def _unit_label(source_name: str, prompt: str | None) -> str:
    """Return one status label for a source and optional batch prompt."""

    return source_name if prompt is None else f"{source_name} -> {prompt}"


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
