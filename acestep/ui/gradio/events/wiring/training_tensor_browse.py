"""Shared tensor-folder browse actions for training tabs."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.training.path_inputs import normalize_user_path
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


def browse_and_load_lora_training_dataset(current_path: str) -> tuple[Any, str, str]:
    """Pick and load a LoRA tensor folder, returning loaded-dataset state."""

    selected = select_optional_folder_path(current_path)
    if selected is None:
        return gr.update(), "No tensor folder selected.", ""

    status, loaded_dir = load_lora_training_dataset_with_state(selected)
    return gr.update(value=selected), status, loaded_dir


def load_lora_training_dataset_with_state(tensor_dir: str) -> tuple[str, str]:
    """Load a LoRA tensor folder and return status plus active dataset path."""

    status = train_h.load_training_dataset(tensor_dir)
    loaded_dir = normalize_user_path(tensor_dir) if _dataset_load_succeeded(status) else ""
    return status, loaded_dir


def _dataset_load_succeeded(status: Any) -> bool:
    """Return whether ``load_training_dataset`` reported a usable tensor set."""

    text = str(status or "")
    return "Loaded preprocessed dataset" in text or (
        "Found" in text and "tensor files" in text
    )
