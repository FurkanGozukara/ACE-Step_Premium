"""Batch Extract runtime for the Advanced generation tab."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from loguru import logger

from acestep.core.generation.cancellation import (
    CANCEL_MESSAGE,
    GenerationCancelled,
    check_generation_cancelled,
    cleanup_runtime_memory,
    generation_cancel_scope,
    is_generation_cancelled,
)
from acestep.ui.gradio.events.batch_folder_args import (
    AUDIO_DURATION_ARG_INDEX,
    AUTOGEN_ARG_INDEX,
    BATCH_QUEUE_ARG_INDEX,
    CURRENT_BATCH_INDEX_ARG_INDEX,
    GENERATION_PARAMS_STATE_ARG_INDEX,
    TOTAL_BATCHES_ARG_INDEX,
    extract_generation_paths,
    extract_generation_status,
)
from acestep.ui.gradio.events.batch_extract_files import (
    audio_duration_seconds,
    copy_batch_extract_audio_outputs,
    discover_batch_extract_audio_files,
    resolve_batch_extract_output_folder,
)


GenerationRunner = Callable[..., Iterable[tuple[Any, ...]]]

BATCH_SIZE_ARG_INDEX = 12
SRC_AUDIO_ARG_INDEX = 13
TASK_TYPE_ARG_INDEX = 20
TRACK_NAME_ARG_INDEX = 55


def _default_generation_runner() -> GenerationRunner:
    """Return the standard generation runner used by single-song Extract."""

    from acestep.ui.gradio.events.results.batch_management import (
        generate_with_batch_management,
    )

    return generate_with_batch_management


def _render_status(lines: Sequence[str]) -> str:
    """Return a bounded status log for the Gradio textbox."""

    if len(lines) <= 80:
        return "\n".join(lines)
    return "\n".join(["... earlier messages omitted ...", *lines[-79:]])


def _validate_extract_settings(generation_args: Sequence[Any]) -> str:
    """Validate the current UI settings and return the selected track name."""

    if len(generation_args) <= TRACK_NAME_ARG_INDEX:
        raise ValueError("Batch Extract received incomplete generation settings.")
    if generation_args[TASK_TYPE_ARG_INDEX] != "extract":
        raise ValueError("Switch Generation Mode to Extract before starting Batch Extract.")
    track_name = str(generation_args[TRACK_NAME_ARG_INDEX] or "").strip()
    if not track_name:
        raise ValueError("Select Track Name before starting Batch Extract.")
    return track_name


def _build_extract_args_for_file(
    generation_args: Sequence[Any],
    audio_path: Path,
    track_name: str,
) -> list[Any]:
    """Return generation args adjusted for one source audio file."""

    args = list(generation_args)
    args[0] = track_name
    args[1] = ""
    args[AUDIO_DURATION_ARG_INDEX] = audio_duration_seconds(audio_path)
    args[BATCH_SIZE_ARG_INDEX] = 1
    args[SRC_AUDIO_ARG_INDEX] = str(audio_path)
    args[TASK_TYPE_ARG_INDEX] = "extract"
    args[TRACK_NAME_ARG_INDEX] = track_name
    args[AUTOGEN_ARG_INDEX] = False
    args[CURRENT_BATCH_INDEX_ARG_INDEX] = 0
    args[TOTAL_BATCHES_ARG_INDEX] = 1
    args[BATCH_QUEUE_ARG_INDEX] = {}
    args[GENERATION_PARAMS_STATE_ARG_INDEX] = {}
    return args


def _run_one_extract(
    runner: GenerationRunner,
    dit_handler: Any,
    llm_handler: Any,
    args: Sequence[Any],
) -> tuple[list[str], str]:
    """Run one Extract request and return generated paths plus final status."""

    final_result: tuple[Any, ...] | None = None
    final_status = ""
    for partial in runner(dit_handler, llm_handler, *args):
        check_generation_cancelled()
        final_result = partial
        partial_status = extract_generation_status(partial)
        if partial_status:
            final_status = partial_status
    check_generation_cancelled()
    return extract_generation_paths(final_result), final_status


def run_batch_extract_processing(
    dit_handler: Any,
    llm_handler: Any,
    input_folder: str,
    output_folder: str,
    generation_args: Sequence[Any],
    *,
    generation_runner: GenerationRunner | None = None,
) -> Iterator[str]:
    """Run Extract for every audio file in a folder and save renamed outputs."""

    status_lines: list[str] = []
    try:
        track_name = _validate_extract_settings(generation_args)
        audio_files = discover_batch_extract_audio_files(input_folder)
        target_folder = resolve_batch_extract_output_folder(output_folder)
    except ValueError as exc:
        yield str(exc)
        return

    runner = generation_runner or _default_generation_runner()
    completed = 0
    started_at = time.time()
    with generation_cancel_scope():
        try:
            status_lines.append(f"Found {len(audio_files)} audio file(s).")
            status_lines.append(f"Extracting track: {track_name}")
            status_lines.append(f"Saving extracted files to: {target_folder}")
            yield _render_status(status_lines)

            for index, audio_path in enumerate(audio_files, start=1):
                check_generation_cancelled()
                status_lines.append(f"[{index}/{len(audio_files)}] Extracting {audio_path.name}")
                yield _render_status(status_lines)
                item_args = _build_extract_args_for_file(generation_args, audio_path, track_name)
                try:
                    generated_paths, item_status = _run_one_extract(
                        runner,
                        dit_handler,
                        llm_handler,
                        item_args,
                    )
                    copied_paths = copy_batch_extract_audio_outputs(
                        generated_paths,
                        audio_path,
                        target_folder,
                    )
                except GenerationCancelled:
                    raise
                except Exception as exc:
                    logger.exception("[batch_extract] Extract failed for {}", audio_path)
                    status_lines.append(f"[{index}/{len(audio_files)}] Failed: {exc}")
                    yield _render_status(status_lines)
                    continue

                if copied_paths:
                    completed += 1
                    saved_list = ", ".join(copied_paths)
                    status_lines.append(f"[{index}/{len(audio_files)}] Saved: {saved_list}")
                else:
                    status_lines.append(
                        f"[{index}/{len(audio_files)}] No extracted audio returned. {item_status}"
                    )
                yield _render_status(status_lines)
        except GenerationCancelled:
            cleanup_runtime_memory()
            status_lines.append(CANCEL_MESSAGE)
            status_lines.append("Batch Extract cancelled. Remaining files were not started.")
            yield _render_status(status_lines)
            return
        finally:
            if is_generation_cancelled():
                cleanup_runtime_memory()

    elapsed = round(max(0.0, time.time() - started_at), 1)
    status_lines.append(
        f"Batch Extract complete: {completed}/{len(audio_files)} file(s) saved in {elapsed}s."
    )
    yield _render_status(status_lines)
