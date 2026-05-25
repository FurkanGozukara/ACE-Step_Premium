"""Single-job execution helpers for Grid Testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from acestep.core.generation.cancellation import check_generation_cancelled
from acestep.ui.gradio.events.batch_folder_args import (
    extract_generation_paths,
    extract_generation_status,
)
from acestep.ui.gradio.events.grid_testing_loras import GridLoraJob
from acestep.ui.gradio.events.grid_testing_status import (
    emit_grid_status,
    grid_files_update,
    render_grid_status,
)
from acestep.ui.gradio.events.results.output_manager import (
    use_generation_run_name,
    use_results_dir,
)


GenerationRunner = Callable[..., Iterable[tuple[Any, ...]]]


def run_grid_job_stream(
    runner: GenerationRunner,
    dit_handler: Any,
    llm_handler: Any,
    args: Sequence[Any],
    temp_root: Path,
    job: GridLoraJob,
    status_lines: list[str],
    final_paths: list[str],
) -> Iterator[tuple[str, Any]]:
    """Run one LoRA job in a temporary generation folder.

    Yields:
        Gradio status/file updates while the underlying generation streams.

    Returns:
        Tuple of generated artifact paths and final generation status.
    """

    final_result: tuple[Any, ...] | None = None
    final_status = ""
    for partial in _stream_with_generation_output_scope(
        runner(dit_handler, llm_handler, *args),
        temp_root=temp_root,
        run_name=job.prefix,
    ):
        check_generation_cancelled()
        final_result = partial
        partial_status = extract_generation_status(partial)
        if partial_status:
            final_status = partial_status
            emit_grid_status(status_lines, f"{job.label}: {partial_status}")
            yield render_grid_status(status_lines), grid_files_update(final_paths)

    generated_paths = extract_generation_paths(final_result)
    if not generated_paths:
        raise RuntimeError(final_status or f"No output paths returned for {job.label}.")
    return generated_paths, final_status or "Generation Complete"


def _stream_with_generation_output_scope(
    stream: Iterable[tuple[Any, ...]],
    *,
    temp_root: Path,
    run_name: str,
) -> Iterator[tuple[Any, ...]]:
    """Resume a generation stream with grid output overrides active.

    Gradio can resume synchronous streaming generators under different
    ``contextvars`` contexts.  The output override must therefore be entered
    and exited around each inner ``next()`` call, not around the whole loop.
    """

    iterator = iter(stream)
    while True:
        with use_results_dir(temp_root), use_generation_run_name(run_name):
            try:
                partial = next(iterator)
            except StopIteration:
                return
        yield partial


def grid_manifest_row(
    job: GridLoraJob,
    status: str,
    output_paths: list[str],
) -> dict[str, Any]:
    """Return one Grid Testing manifest row."""

    return {
        "lora_label": job.label,
        "lora_path": job.path,
        "prefix": job.prefix,
        "status": status,
        "output_paths": output_paths,
    }


def grid_job_start_message(
    job_index: int,
    total_jobs: int,
    job: GridLoraJob,
    processed: int,
    total_songs: int,
) -> str:
    """Return the status line for a LoRA job start."""

    remaining = max(0, total_songs - processed)
    return (
        f"[{job_index}/{total_jobs}] Processing {job.label} "
        f"({processed} processed, {remaining} left)."
    )
