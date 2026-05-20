"""Lazy facade for training event handlers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from acestep.ui.gradio.events.training.training_utils import SAFE_TRAINING_ROOT


def _load_attr(module_name: str, attr_name: str) -> Any:
    """Import and return the requested training attribute."""

    module = import_module(f"acestep.ui.gradio.events.training.{module_name}")
    return getattr(module, attr_name)


def _forward(module_name: str, attr_name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve a training handler lazily and invoke it."""

    return _load_attr(module_name, attr_name)(*args, **kwargs)


def create_dataset_builder(*args: Any, **kwargs: Any) -> Any:
    return _forward("training_utils", "create_dataset_builder", *args, **kwargs)


def _safe_slider(*args: Any, **kwargs: Any) -> Any:
    return _forward("training_utils", "_safe_slider", *args, **kwargs)


def _safe_join(*args: Any, **kwargs: Any) -> Any:
    return _forward("training_utils", "_safe_join", *args, **kwargs)


def _format_duration(*args: Any, **kwargs: Any) -> Any:
    return _forward("training_utils", "_format_duration", *args, **kwargs)


def _training_loss_figure(*args: Any, **kwargs: Any) -> Any:
    return _forward("training_utils", "_training_loss_figure", *args, **kwargs)


def scan_directory(*args: Any, **kwargs: Any) -> Any:
    return _forward("dataset_ops", "scan_directory", *args, **kwargs)


def auto_label_all(*args: Any, **kwargs: Any) -> Any:
    return _forward("dataset_ops", "auto_label_all", *args, **kwargs)


def get_sample_preview(*args: Any, **kwargs: Any) -> Any:
    return _forward("dataset_ops", "get_sample_preview", *args, **kwargs)


def save_sample_edit(*args: Any, **kwargs: Any) -> Any:
    return _forward("dataset_ops", "save_sample_edit", *args, **kwargs)


def update_settings(*args: Any, **kwargs: Any) -> Any:
    return _forward("dataset_ops", "update_settings", *args, **kwargs)


def save_dataset(*args: Any, **kwargs: Any) -> Any:
    return _forward("dataset_ops", "save_dataset", *args, **kwargs)


def load_existing_dataset_for_preprocess(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "preprocess",
        "load_existing_dataset_for_preprocess",
        *args,
        **kwargs,
    )


def preprocess_dataset(*args: Any, **kwargs: Any) -> Any:
    return _forward("preprocess", "preprocess_dataset", *args, **kwargs)


def load_training_dataset(*args: Any, **kwargs: Any) -> Any:
    return _forward("preprocess", "load_training_dataset", *args, **kwargs)


def start_training(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_training", "start_training", *args, **kwargs)


def stop_training(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_training", "stop_training", *args, **kwargs)


def open_lora_output_folder(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_training", "open_lora_output_folder", *args, **kwargs)


def export_lora(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_training", "export_lora", *args, **kwargs)


def start_lokr_training(*args: Any, **kwargs: Any) -> Any:
    return _forward("lokr_training", "start_lokr_training", *args, **kwargs)


def list_lokr_export_epochs(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "lokr_training",
        "list_lokr_export_epochs",
        *args,
        **kwargs,
    )


def export_lokr(*args: Any, **kwargs: Any) -> Any:
    return _forward("lokr_training", "export_lokr", *args, **kwargs)


__all__ = [
    "SAFE_TRAINING_ROOT",
    "create_dataset_builder",
    "_safe_slider",
    "_safe_join",
    "_format_duration",
    "_training_loss_figure",
    "scan_directory",
    "auto_label_all",
    "get_sample_preview",
    "save_sample_edit",
    "update_settings",
    "save_dataset",
    "load_existing_dataset_for_preprocess",
    "preprocess_dataset",
    "load_training_dataset",
    "start_training",
    "stop_training",
    "open_lora_output_folder",
    "export_lora",
    "start_lokr_training",
    "list_lokr_export_epochs",
    "export_lokr",
]
