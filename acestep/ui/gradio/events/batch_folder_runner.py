"""Batch folder processing runtime for the Gradio UI."""

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
    CAPTION_ARG_INDEX,
    build_generation_args_for_item,
    extract_generation_paths,
    extract_generation_status,
)
from acestep.ui.gradio.events.batch_folder_files import (
    BatchFolderItem,
    discover_batch_folder_items,
    resolve_output_folder,
)
from acestep.ui.gradio.events.batch_folder_text import improve_batch_text_if_requested
from acestep.ui.gradio.events.results.output_manager import (
    use_generation_run_name,
    use_results_dir,
    write_json,
)


GenerationRunner = Callable[..., Iterable[tuple[Any, ...]]]


def _default_generation_runner() -> GenerationRunner:
    """Return the standard UI batch-generation runner."""

    from acestep.ui.gradio.events.results.batch_management import (
        generate_with_batch_management,
    )

    return generate_with_batch_management


def _render_status(lines: Sequence[str]) -> str:
    """Return a compact status log for the Gradio textbox."""

    if len(lines) <= 80:
        return "\n".join(lines)
    return "\n".join(["... earlier messages omitted ...", *lines[-79:]])


def _item_caption(item: BatchFolderItem, generation_args: Sequence[Any]) -> str:
    """Return the style prompt for an item, falling back to the current UI caption."""

    if item.style.strip():
        return item.style.strip()
    return str(generation_args[CAPTION_ARG_INDEX] or "").strip()


def _write_manifest(output_folder: Path, rows: list[dict[str, Any]]) -> str:
    """Persist batch-processing results beside the generated run folders."""

    return write_json(
        output_folder / "batch_folder_manifest.json",
        {
            "_meta": {
                "format": "ace_step_batch_folder_manifest",
                "version": 1,
                "updated_at_unix": time.time(),
            },
            "items": rows,
        },
    )


def _run_one_generation(
    runner: GenerationRunner,
    dit_handler: Any,
    llm_handler: Any,
    args: Sequence[Any],
) -> tuple[list[str], str]:
    """Run one generation request and return output paths plus final status."""

    final_result: tuple[Any, ...] | None = None
    final_status = ""
    for partial in runner(dit_handler, llm_handler, *args):
        check_generation_cancelled()
        final_result = partial
        partial_status = extract_generation_status(partial)
        if partial_status:
            final_status = partial_status
    return extract_generation_paths(final_result), final_status


def run_batch_folder_processing(
    dit_handler: Any,
    llm_handler: Any,
    input_folder: str,
    output_folder: str,
    auto_improve_lyrics: bool,
    auto_improve_style: bool,
    generation_args: Sequence[Any],
    *,
    generation_runner: GenerationRunner | None = None,
) -> Iterator[str]:
    """Process lyrics/style text-file pairs into generated audio runs."""

    status_lines: list[str] = []
    status_lines.append("Batch folder processing started. Scanning input folder...")
    yield _render_status(status_lines)
    try:
        items = discover_batch_folder_items(input_folder)
        target_folder = resolve_output_folder(output_folder)
        item_args = list(generation_args)
    except ValueError as exc:
        yield str(exc)
        return

    runner = generation_runner or _default_generation_runner()
    rows: list[dict[str, Any]] = []

    with generation_cancel_scope():
        try:
            status_lines.append(f"Found {len(items)} lyrics file(s).")
            status_lines.append(f"Saving batch outputs under: {target_folder}")
            yield _render_status(status_lines)

            for index, item in enumerate(items, start=1):
                check_generation_cancelled()
                status_lines.append(f"[{index}/{len(items)}] Processing {item.lyrics_path.name}")
                yield _render_status(status_lines)

                caption = _item_caption(item, item_args)
                text_result = improve_batch_text_if_requested(
                    llm_handler,
                    item_args,
                    caption=caption,
                    lyrics=item.lyrics,
                    improve_style=bool(auto_improve_style),
                    improve_lyrics=bool(auto_improve_lyrics),
                )
                check_generation_cancelled()
                if text_result.status:
                    status_lines.extend(text_result.status.splitlines())
                    yield _render_status(status_lines)

                generated_paths: list[str] = []
                item_status = "Generation did not return output paths."
                started_at = time.time()
                try:
                    generation_call_args = build_generation_args_for_item(
                        item_args,
                        caption=text_result.caption,
                        lyrics=text_result.lyrics,
                        is_formatted=text_result.formatted,
                    )
                    with use_results_dir(target_folder), use_generation_run_name(item.stem):
                        generated_paths, item_status = _run_one_generation(
                            runner,
                            dit_handler,
                            llm_handler,
                            generation_call_args,
                        )
                    check_generation_cancelled()
                except GenerationCancelled:
                    raise
                except Exception as exc:
                    logger.exception("[batch_folder] Generation failed for {}", item.lyrics_path)
                    item_status = f"Failed: {exc}"

                row = {
                    "name": item.stem,
                    "lyrics_path": str(item.lyrics_path),
                    "style_path": str(item.style_path) if item.style_path else None,
                    "status": "completed" if generated_paths else "failed",
                    "message": item_status,
                    "duration_seconds": round(max(0.0, time.time() - started_at), 3),
                    "output_paths": generated_paths,
                }
                rows.append(row)
                manifest_path = _write_manifest(target_folder, rows)

                if generated_paths:
                    status_lines.append(f"[{index}/{len(items)}] Done: {item.stem}")
                else:
                    status_lines.append(f"[{index}/{len(items)}] {item_status}")
                status_lines.append(f"Manifest: {manifest_path}")
                yield _render_status(status_lines)
        except GenerationCancelled:
            cleanup_runtime_memory()
            status_lines.append(CANCEL_MESSAGE)
            status_lines.append("Batch cancelled. Remaining files were not started.")
            yield _render_status(status_lines)
            return
        finally:
            if is_generation_cancelled():
                cleanup_runtime_memory()

    completed = sum(1 for row in rows if row["status"] == "completed")
    status_lines.append(f"Batch complete: {completed}/{len(items)} item(s) generated.")
    yield _render_status(status_lines)
