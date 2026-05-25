"""Output contracts for Dataset page import and preview wiring."""

from __future__ import annotations

from typing import Any

import gradio as gr


def dataset_import_outputs(dataset_section: dict[str, Any]) -> list[Any]:
    """Return output components updated by Dataset page imports."""

    return [
        *dataset_preview_outputs(dataset_section),
        dataset_section["ngram_browser_group"],
        dataset_section["ngram_1_table"],
        dataset_section["ngram_2_table"],
        dataset_section["ngram_3_table"],
        dataset_section["ngram_4_table"],
        dataset_section["ngram_5_table"],
        dataset_section["ngram_6_table"],
        dataset_section["selected_ngram_display"],
        dataset_section["ngram_song_table"],
        dataset_section["selected_ngram"],
        dataset_section["selected_ngram_size"],
        dataset_section["get_item_btn"],
    ]


def dataset_preview_outputs(dataset_section: dict[str, Any]) -> list[Any]:
    """Return output components updated by item preview events."""

    return [
        dataset_section["data_status"],
        dataset_section["instruction_display"],
        dataset_section["item_info_json"],
        dataset_section["item_src_audio"],
        dataset_section["item_target_audio"],
        dataset_section["item_refer_audio"],
    ]


def finish_dataset_import(dataset_handler: Any, result: tuple[Any, ...]) -> tuple[Any, ...]:
    """Add visibility and Get Item interactivity to Dataset page import outputs."""

    values = tuple(result)
    preview_values = values[:6] if len(values) >= 6 else empty_preview_values()
    if len(values) >= 16:
        ngram_values = values[6:]
        ngram_outputs = (
            *[_ngram_choices_update(rows) for rows in ngram_values[:6]],
            ngram_values[6],
            gr.update(choices=[], value=None, interactive=False),
            ngram_values[8],
            ngram_values[9],
        )
    else:
        ngram_outputs = empty_ngram_values()
    imported = bool(dataset_handler.dataset_imported)
    return (
        *preview_values,
        gr.update(visible=imported),
        *ngram_outputs,
        gr.update(interactive=imported),
    )


def empty_dataset_import(status: str) -> tuple[Any, ...]:
    """Return empty Dataset page import updates with a visible status."""

    return (
        status,
        *empty_preview_values()[1:],
        gr.update(visible=False),
        *empty_ngram_values(),
        gr.update(interactive=False),
    )


def pending_dataset_import(status: str) -> tuple[Any, ...]:
    """Return no-op Dataset page import updates with progress status text."""

    return (status, *[gr.update() for _ in range(len(DATASET_IMPORT_OUTPUT_KEYS) - 1)])


def empty_preview_values() -> tuple[Any, ...]:
    """Return empty preview update values for Dataset page outputs."""

    return ("No dataset imported.", *[gr.update() for _ in range(5)])


def empty_ngram_values() -> tuple[Any, ...]:
    """Return empty n-gram browser values for Dataset page outputs."""

    return (
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        "Import a dataset JSON to populate the gram columns.",
        gr.update(choices=[], value=None, interactive=False),
        "",
        0,
    )


def _ngram_choices_update(rows: Any) -> dict[str, Any]:
    """Return a Radio update for n-gram rows."""

    return gr.update(choices=_ngram_choices(rows), value=None)


def _ngram_choices(rows: Any) -> list[tuple[str, str]]:
    """Return Radio choices from n-gram table rows."""

    choices: list[tuple[str, str]] = []
    for row in rows or []:
        gram, songs, _hits = row
        choices.append((f"{gram}  |  {songs} songs", gram))
    return choices


DATASET_IMPORT_OUTPUT_KEYS = (
    "data_status",
    "instruction_display",
    "item_info_json",
    "item_src_audio",
    "item_target_audio",
    "item_refer_audio",
    "ngram_browser_group",
    "ngram_1_table",
    "ngram_2_table",
    "ngram_3_table",
    "ngram_4_table",
    "ngram_5_table",
    "ngram_6_table",
    "selected_ngram_display",
    "ngram_song_table",
    "selected_ngram",
    "selected_ngram_size",
    "get_item_btn",
)
