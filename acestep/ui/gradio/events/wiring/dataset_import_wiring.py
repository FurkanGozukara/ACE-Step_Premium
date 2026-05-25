"""Dataset Explorer import and item-preview wiring."""

from __future__ import annotations

import time
from typing import Any

import gradio as gr
from loguru import logger

from acestep.dataset_ngrams import song_rows_for_ngram

from .context import GenerationWiringContext
from .dataset_import_outputs import (
    dataset_import_outputs,
    dataset_preview_outputs,
    empty_dataset_import,
    finish_dataset_import,
    pending_dataset_import,
)


def register_dataset_import_handlers(context: GenerationWiringContext) -> None:
    """Register Dataset page import and preview handlers."""

    dataset_section = context.dataset_section
    dataset_handler = context.dataset_handler
    import_outputs = dataset_import_outputs(dataset_section)

    def import_dataset_for_page(
        dataset_type: str,
        dataset_path: str,
        progress=gr.Progress(),
    ):
        """Import a JSON/folder path and update Dataset page preview outputs."""

        progress(0.05, desc="Loading dataset JSON")
        yield pending_dataset_import("0% - Loading dataset JSON...")
        start_time = time.perf_counter()
        logger.info(f"Dataset Explorer import started: {dataset_path}")
        result = dataset_handler.import_dataset_for_ui(dataset_type, dataset_path)
        progress(0.85, desc="Building caption/style word grams")
        yield pending_dataset_import("75% - Building caption/style word grams...")
        outputs = finish_dataset_import(dataset_handler, result)
        progress(1.0, desc="Dataset ready")
        elapsed = time.perf_counter() - start_time
        logger.info(f"Dataset Explorer import finished in {elapsed:.2f}s")
        yield outputs

    def import_selected_json_file(
        dataset_type: str,
        selected_file: Any,
        progress=gr.Progress(),
    ):
        """Import a browser-selected dataset JSON file."""

        selected_path = _file_value_path(selected_file)
        if not selected_path:
            yield (gr.update(), *empty_dataset_import("No dataset JSON selected."))
            return
        progress(0.05, desc="Loading selected dataset JSON")
        yield (
            gr.update(value=selected_path),
            *pending_dataset_import("0% - Loading selected dataset JSON..."),
        )
        start_time = time.perf_counter()
        logger.info(f"Dataset Explorer import started: {selected_path}")
        result = dataset_handler.import_dataset_for_ui(dataset_type, selected_path)
        progress(0.85, desc="Building caption/style word grams")
        yield (
            gr.update(value=selected_path),
            *pending_dataset_import("75% - Building caption/style word grams..."),
        )
        outputs = finish_dataset_import(dataset_handler, result)
        progress(1.0, desc="Dataset ready")
        elapsed = time.perf_counter() - start_time
        logger.info(f"Dataset Explorer import finished in {elapsed:.2f}s")
        yield (gr.update(value=selected_path), *outputs)

    dataset_section["dataset_json_file"].change(
        fn=import_selected_json_file,
        inputs=[
            dataset_section["dataset_type"],
            dataset_section["dataset_json_file"],
        ],
        outputs=[dataset_section["dataset_import_path"], *import_outputs],
    )

    dataset_section["import_dataset_btn"].click(
        fn=import_dataset_for_page,
        inputs=[
            dataset_section["dataset_type"],
            dataset_section["dataset_import_path"],
        ],
        outputs=import_outputs,
    )
    dataset_section["get_item_btn"].click(
        fn=dataset_handler.get_item_for_ui,
        inputs=[
            dataset_section["search_type"],
            dataset_section["search_value"],
        ],
        outputs=dataset_preview_outputs(dataset_section),
    )

    for gram_size, table_key in (
        (1, "ngram_1_table"),
        (2, "ngram_2_table"),
        (3, "ngram_3_table"),
        (4, "ngram_4_table"),
        (5, "ngram_5_table"),
        (6, "ngram_6_table"),
    ):
        dataset_section[table_key].change(
            fn=_make_ngram_value_handler(dataset_handler, gram_size),
            inputs=[dataset_section[table_key]],
            outputs=[
                dataset_section["selected_ngram_display"],
                dataset_section["ngram_song_table"],
                dataset_section["selected_ngram"],
                dataset_section["selected_ngram_size"],
            ],
        )

    dataset_section["ngram_song_table"].change(
        fn=_make_ngram_song_value_handler(dataset_handler),
        inputs=[dataset_section["ngram_song_table"]],
        outputs=dataset_preview_outputs(dataset_section),
    )


def _make_ngram_value_handler(dataset_handler: Any, gram_size: int):
    """Build a gram selector callback for one n-gram size."""

    def select_ngram(gram: str) -> tuple[Any, ...]:
        """Return songs matching the selected n-gram."""

        if not gram:
            return (
                "Select a gram from the columns above.",
                gr.update(choices=[], value=None, interactive=False),
                "",
                0,
            )
        rows = song_rows_for_ngram(dataset_handler.dataset, gram_size, gram)
        choices = [
            (
                f"{row[1]}  |  {row[3]}  |  {row[4]}",
                str(row[0]),
            )
            for row in rows
        ]
        return (
            f"{gram_size}-gram: {gram} | {len(rows)} song(s)",
            gr.update(choices=choices, value=None, interactive=bool(choices)),
            gram,
            int(gram_size),
        )

    return select_ngram


def _make_ngram_song_value_handler(dataset_handler: Any):
    """Build a matching-song selector callback."""

    def select_song(sample_index: str) -> tuple[Any, ...]:
        """Return preview data for the selected matching song."""

        if sample_index in (None, ""):
            return ("Select a song from the gram matches.", "", "{}", None, None, None)
        return dataset_handler.get_item_for_ui("idx", str(sample_index))

    return select_song


def _file_value_path(value: Any) -> str:
    """Return a filepath from a Gradio File value."""

    if isinstance(value, (list, tuple)):
        return _file_value_path(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or "")
    return str(value or "")
