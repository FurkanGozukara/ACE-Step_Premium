"""Gradio helpers for LoRA training VRAM preset controls."""

from __future__ import annotations

import gradio as gr

from acestep.training.lora_vram_presets import (
    DEFAULT_LORA_VRAM_PRESET,
    LORA_VRAM_PRESET_CHOICES,
    LORA_VRAM_PRESET_MANUAL,
    default_lora_vram_preset_name,
    get_lora_vram_preset,
)


def default_lora_vram_control_values() -> dict[str, object]:
    """Return initial LoRA VRAM control values from the selected default preset."""

    return get_lora_vram_preset(default_lora_vram_preset_name())


def build_lora_vram_preset_dropdown() -> gr.Dropdown:
    """Render the LoRA VRAM preset selector."""

    return gr.Dropdown(
        label="VRAM Preset",
        choices=LORA_VRAM_PRESET_CHOICES,
        value=default_lora_vram_preset_name(),
        info="Applies measured memory settings for batch size 1 and grad accumulation 1.",
        elem_classes=["has-info-container"],
    )


def lora_vram_preset_updates(preset_name: str):
    """Return Gradio updates for controls managed by a preset selection."""

    preset = get_lora_vram_preset(preset_name)
    if not preset:
        return (gr.update(),) * 10

    return (
        gr.update(value=preset["lora_rank"]),
        gr.update(value=preset["lora_alpha"]),
        gr.update(value=preset["gradient_checkpointing"]),
        gr.update(value=preset["activation_cpu_offload"]),
        gr.update(value=preset["offload_non_decoder"]),
        gr.update(value=preset["keep_frozen_base_in_compute_dtype"]),
        gr.update(value=preset.get("optimizer_type", "adamw8bit")),
        gr.update(value=preset.get("scheduler_type", "constant")),
        gr.update(value=preset["base_quantization"]),
        gr.update(value=preset["empty_cache_every_n_steps"]),
    )


__all__ = [
    "DEFAULT_LORA_VRAM_PRESET",
    "LORA_VRAM_PRESET_MANUAL",
    "build_lora_vram_preset_dropdown",
    "default_lora_vram_control_values",
    "default_lora_vram_preset_name",
    "lora_vram_preset_updates",
]
