"""Preset and stage-toggle actions for the Audio Processing page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.presets import (
    PROCESSING_PRESET_NONE,
    PRESET_VALUES,
    STAGE_KEYS,
)


def apply_builtin_preset(preset_name: str | None) -> tuple[Any, ...]:
    """Return stage value and enabled updates for a built-in processing preset."""

    selected = str(preset_name or "")
    values = PRESET_VALUES.get(selected, PRESET_VALUES["Generic AI"])
    enabled = selected != PROCESSING_PRESET_NONE
    value_updates = tuple(gr.update(value=values[key]) for key in STAGE_KEYS)
    enabled_updates = tuple(gr.update(value=enabled) for _key in STAGE_KEYS)
    return value_updates + enabled_updates


def toggle_audio_enhancement_stages(*enabled_values: Any) -> tuple[Any, ...]:
    """Return updates that toggle all stage checkboxes."""

    target = not all(bool(value) for value in enabled_values)
    return tuple(gr.update(value=target) for _ in STAGE_KEYS)
