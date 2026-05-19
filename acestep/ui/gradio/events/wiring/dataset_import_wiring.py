"""Dataset Explorer import and item-preview wiring."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ..local_path_dialogs import (
    normalize_dialog_path,
    select_folder_path,
    select_json_file_path,
)
from .context import GenerationWiringContext


def register_dataset_import_handlers(context: GenerationWiringContext) -> None:
    """Register Dataset page import and preview handlers."""

    dataset_section = context.dataset_section
    dataset_handler = context.dataset_handler
    import_outputs = _dataset_import_outputs(dataset_section)

    def import_dataset_for_page(dataset_type: str, dataset_path: str) -> tuple[Any, ...]:
        """Import a JSON/folder path and update Dataset page preview outputs."""

        result = dataset_handler.import_dataset_for_ui(dataset_type, dataset_path)
        return _finish_dataset_import(dataset_handler, result)

    def browse_and_import_json(dataset_type: str, current_path: str) -> tuple[Any, ...]:
        """Pick a dataset JSON path and import it into the Dataset page."""

        selected = select_json_file_path(current_path)
        if not selected or selected == normalize_dialog_path(current_path):
            return (gr.update(), *_empty_dataset_import("No dataset JSON selected."))
        result = dataset_handler.import_dataset_for_ui(dataset_type, selected)
        return (gr.update(value=selected), *_finish_dataset_import(dataset_handler, result))

    def browse_and_import_folder(dataset_type: str, current_path: str) -> tuple[Any, ...]:
        """Pick an audio folder path and import it into the Dataset page."""

        selected = select_folder_path(current_path)
        if not selected or selected == normalize_dialog_path(current_path):
            return (gr.update(), *_empty_dataset_import("No dataset folder selected."))
        result = dataset_handler.import_dataset_for_ui(dataset_type, selected)
        return (gr.update(value=selected), *_finish_dataset_import(dataset_handler, result))

    dataset_section["import_json_browse_btn"].click(
        fn=browse_and_import_json,
        inputs=[
            dataset_section["dataset_type"],
            dataset_section["dataset_import_path"],
        ],
        outputs=[dataset_section["dataset_import_path"], *import_outputs],
    )
    dataset_section["import_folder_browse_btn"].click(
        fn=browse_and_import_folder,
        inputs=[
            dataset_section["dataset_type"],
            dataset_section["dataset_import_path"],
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
        outputs=import_outputs[:-1],
    )


def _dataset_import_outputs(dataset_section: dict[str, Any]) -> list[Any]:
    """Return output components updated by Dataset page imports."""

    return [
        dataset_section["data_status"],
        dataset_section["instruction_display"],
        dataset_section["item_info_json"],
        dataset_section["item_src_audio"],
        dataset_section["item_target_audio"],
        dataset_section["item_refer_audio"],
        dataset_section["get_item_btn"],
    ]


def _finish_dataset_import(dataset_handler: Any, result: tuple[Any, ...]) -> tuple[Any, ...]:
    """Append Get Item interactivity to Dataset page import outputs."""

    return (*result, gr.update(interactive=bool(dataset_handler.dataset_imported)))


def _empty_dataset_import(status: str) -> tuple[Any, ...]:
    """Return no-op Dataset page preview updates with a visible status."""

    return (
        status,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )
