"""Runtime orchestration for the Grid Testing tab."""

from __future__ import annotations

import shutil
import time
from typing import Any, Callable, Iterable, Iterator, Sequence

from loguru import logger

from acestep.core.generation.cancellation import (
    CANCEL_MESSAGE,
    GenerationCancelled,
    cleanup_runtime_memory,
    generation_cancel_scope,
    is_generation_cancelled,
)
from acestep.ui.gradio.events.grid_testing_args import (
    apply_grid_generation_count,
    apply_grid_seed,
    args_for_grid_lora,
    prepare_grid_generation_args,
)
from acestep.ui.gradio.events.grid_testing_files import (
    flatten_generation_outputs,
    write_grid_manifest,
)
from acestep.ui.gradio.events.grid_testing_job import (
    grid_job_start_message,
    grid_manifest_row,
    run_grid_job_stream,
)
from acestep.ui.gradio.events.grid_testing_loras import resolve_grid_lora_jobs
from acestep.ui.gradio.events.grid_testing_paths import resolve_grid_output_folder
from acestep.ui.gradio.events.grid_testing_status import (
    emit_grid_status,
    grid_files_update,
    render_grid_status,
)


GenerationRunner = Callable[..., Iterable[tuple[Any, ...]]]


def run_grid_testing(
    dit_handler: Any,
    llm_handler: Any,
    selected_loras: Any,
    output_folder: str,
    mp3_only: bool,
    generation_args: Sequence[Any],
    *,
    generations_per_lora: Any = 1,
    generation_runner: GenerationRunner | None = None,
) -> Iterator[tuple[str, Any]]:
    """Generate the current quick settings for each selected LoRA.

    Args:
        dit_handler: Active DiT handler.
        llm_handler: Active language-model handler.
        selected_loras: Raw Grid Testing LoRA multiselect value.
        output_folder: Optional custom final output folder.
        mp3_only: Whether to keep only MP3 outputs.
        generation_args: Current generation settings from the quick/advanced controls.
        generations_per_lora: Number of examples to generate for each selected LoRA.
        generation_runner: Optional replacement for the standard generation runner.

    Yields:
        ``(status_text, files_update)`` tuples for Gradio.
    """

    status_lines: list[str] = []
    final_paths: list[str] = []
    emit_grid_status(status_lines, "Grid Testing started. Preparing selected LoRAs...")
    yield render_grid_status(status_lines), grid_files_update(final_paths)
    try:
        args = prepare_grid_generation_args(generation_args, mp3_only=bool(mp3_only))
        generation_count = apply_grid_generation_count(args, generations_per_lora)
        jobs = resolve_grid_lora_jobs(selected_loras)
        target_folder = resolve_grid_output_folder(output_folder)
    except ValueError as exc:
        yield str(exc), grid_files_update([])
        return

    runner = generation_runner or _default_generation_runner()
    temp_parent = target_folder / ".grid_work"
    temp_root = temp_parent / f"{int(time.time() * 1000)}"
    rows: list[dict[str, Any]] = []
    total_songs = len(jobs) * generation_count

    with generation_cancel_scope():
        try:
            seed_status = apply_grid_seed(args)
            if seed_status:
                emit_grid_status(status_lines, seed_status)
            emit_grid_status(status_lines, f"Grid output folder: {target_folder}")
            emit_grid_status(
                status_lines,
                f"Grid jobs: {len(jobs)} LoRA(s), {total_songs} song(s) total.",
            )
            yield render_grid_status(status_lines), grid_files_update(final_paths)

            processed = 0
            for job_index, job in enumerate(jobs, start=1):
                current_args = args_for_grid_lora(args, job)
                emit_grid_status(
                    status_lines,
                    grid_job_start_message(job_index, len(jobs), job, processed, total_songs),
                )
                yield render_grid_status(status_lines), grid_files_update(final_paths)

                generated_paths, generation_status = yield from run_grid_job_stream(
                    runner,
                    dit_handler,
                    llm_handler,
                    current_args,
                    temp_root,
                    job,
                    status_lines,
                    final_paths,
                )
                flattened = flatten_generation_outputs(
                    generated_paths,
                    target_folder,
                    prefix=job.prefix,
                    caption=current_args[0],
                    lyrics=current_args[1],
                    mp3_only=bool(mp3_only),
                )
                final_paths.extend(flattened)
                processed += generation_count
                rows.append(grid_manifest_row(job, generation_status, flattened))
                if not mp3_only:
                    manifest_path = write_grid_manifest(target_folder, rows)
                    if manifest_path not in final_paths:
                        final_paths.append(manifest_path)
                emit_grid_status(
                    status_lines,
                    f"Done: {job.label}. Processed {processed}/{total_songs}.",
                )
                yield render_grid_status(status_lines), grid_files_update(final_paths)
        except GenerationCancelled:
            cleanup_runtime_memory()
            emit_grid_status(status_lines, CANCEL_MESSAGE)
            yield render_grid_status(status_lines), grid_files_update(final_paths)
            return
        except Exception as exc:
            logger.exception("[grid_testing] Grid generation failed")
            emit_grid_status(status_lines, f"Grid failed: {exc}")
            yield render_grid_status(status_lines), grid_files_update(final_paths)
            return
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            _remove_empty_temp_parent(temp_parent)
            if is_generation_cancelled():
                cleanup_runtime_memory()

    emit_grid_status(status_lines, f"Grid complete. Files saved: {len(final_paths)}")
    yield render_grid_status(status_lines), grid_files_update(final_paths)


def _default_generation_runner() -> GenerationRunner:
    """Return the standard batch-managed generation runner."""

    from acestep.ui.gradio.events.results.batch_management import (
        generate_with_batch_management,
    )

    return generate_with_batch_management


def _remove_empty_temp_parent(temp_parent: Any) -> None:
    """Remove the grid work parent when no temporary runs remain."""

    try:
        temp_parent.rmdir()
    except OSError:
        return
