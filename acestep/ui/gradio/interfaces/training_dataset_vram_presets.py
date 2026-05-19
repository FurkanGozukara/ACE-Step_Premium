"""Dataset action VRAM preset UI helpers."""

from __future__ import annotations

import gradio as gr

from acestep.training.dataset_vram_presets import (
    DATASET_VRAM_PRESET_CHOICES,
    DEFAULT_DATASET_VRAM_PRESET,
    default_dataset_vram_preset_name,
)


def build_dataset_vram_preset_dropdown() -> gr.Dropdown:
    """Render the dataset action VRAM preset dropdown."""

    return gr.Dropdown(
        label="Dataset VRAM Preset",
        choices=DATASET_VRAM_PRESET_CHOICES,
        value=default_dataset_vram_preset_name(),
        info=(
            "Controls Auto-Label and Preprocess model loading. Presets run in an "
            "isolated worker so VRAM is released when the action finishes."
        ),
        elem_classes=["has-info-container"],
    )


__all__ = [
    "DEFAULT_DATASET_VRAM_PRESET",
    "build_dataset_vram_preset_dropdown",
    "default_dataset_vram_preset_name",
]
