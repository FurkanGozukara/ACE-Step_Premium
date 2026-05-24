"""Dataset save wiring for the training dataset-builder tab."""

from __future__ import annotations

from typing import Any, Mapping

import gradio as gr

from acestep.ui.gradio.events.local_path_dialogs import (
    is_dialog_available,
    normalize_dialog_path,
    select_json_save_path,
)

from .. import training_handlers as train_h
from .context import TrainingWiringContext
from .training_dataset_load_outputs import (
    build_dataset_load_outputs,
    no_dataset_load_outputs,
)


_SAVE_SUCCESS = "\u2705"


def _is_successful_save(status: Any) -> bool:
    """Return whether a save status reports a completed dataset write."""

    return _SAVE_SUCCESS in str(status)


def _update_value(update: Any) -> Any:
    """Return a Gradio update value when the callback provided one."""

    if isinstance(update, dict):
        return update.get("value")
    return None


def _save_and_load_outputs(training_section: Mapping[str, Any]) -> list[Any]:
    """Return ordered outputs for save status, path update, and JSON reload."""

    return [
        training_section["save_status"],
        training_section["save_path"],
        training_section["load_existing_dataset_path"],
        *build_dataset_load_outputs(training_section, "load_existing_status"),
    ]


def _save_dataset_and_load_saved_path(
    save_path: str,
    dataset_name: str,
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    genre_ratio: int,
    builder_state: Any,
) -> tuple[Any, ...]:
    """Save the dataset, then load the saved JSON into Step 5 controls."""

    save_status, save_path_update = train_h.save_dataset(
        save_path,
        dataset_name,
        builder_state,
        custom_tag=custom_tag,
        tag_position=tag_position,
        all_instrumental=all_instrumental,
        genre_ratio=genre_ratio,
    )
    saved_path = _update_value(save_path_update)
    if not saved_path or not _is_successful_save(save_status):
        return (
            save_status,
            save_path_update,
            gr.update(),
            *no_dataset_load_outputs("Save did not complete; dataset was not loaded."),
        )

    load_outputs = train_h.load_existing_dataset_for_preprocess(saved_path, builder_state)
    return (
        save_status,
        save_path_update,
        gr.update(value=saved_path),
        *load_outputs,
    )


def _browse_save_dataset_and_load_saved_path(
    save_path: str,
    dataset_name: str,
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    genre_ratio: int,
    builder_state: Any,
) -> tuple[Any, ...]:
    """Pick a JSON save path, save the dataset, and load the saved JSON."""

    selected = select_json_save_path(save_path)
    current = normalize_dialog_path(save_path)
    if not selected or (is_dialog_available() and selected == current):
        return (
            "No save path selected.",
            gr.update(),
            gr.update(),
            *no_dataset_load_outputs("No save path selected."),
        )
    return _save_dataset_and_load_saved_path(
        selected,
        dataset_name,
        custom_tag,
        tag_position,
        all_instrumental,
        genre_ratio,
        builder_state,
    )


def register_training_dataset_save_handlers(context: TrainingWiringContext) -> None:
    """Register dataset save events and reload the saved JSON on success."""

    training_section = context.training_section
    outputs = _save_and_load_outputs(training_section)

    training_section["save_dataset_btn"].click(
        fn=_save_dataset_and_load_saved_path,
        inputs=[
            training_section["save_path"],
            training_section["dataset_name"],
            training_section["custom_tag"],
            training_section["tag_position"],
            training_section["all_instrumental"],
            training_section["genre_ratio"],
            training_section["dataset_builder_state"],
        ],
        outputs=outputs,
    )

    training_section["save_path_browse_btn"].click(
        fn=_browse_save_dataset_and_load_saved_path,
        inputs=[
            training_section["save_path"],
            training_section["dataset_name"],
            training_section["custom_tag"],
            training_section["tag_position"],
            training_section["all_instrumental"],
            training_section["genre_ratio"],
            training_section["dataset_builder_state"],
        ],
        outputs=outputs,
    )
