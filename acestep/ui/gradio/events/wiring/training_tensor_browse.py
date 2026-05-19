"""Shared tensor-folder browse actions for training tabs."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.events.local_path_dialogs import select_optional_folder_path

from .. import training_handlers as train_h


def browse_and_load_training_dataset(current_path: str) -> tuple[Any, str]:
    """Pick a tensor folder and return textbox update plus dataset summary.

    Args:
        current_path: Current value in the tensor directory textbox.

    Returns:
        A textbox update and the loaded dataset summary or cancellation status.
    """

    selected = select_optional_folder_path(current_path)
    if selected is None:
        return gr.update(), "No tensor folder selected."
    return gr.update(value=selected), train_h.load_training_dataset(selected)
