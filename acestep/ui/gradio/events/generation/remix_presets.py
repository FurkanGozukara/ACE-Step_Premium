"""Remix preset helpers for paired strength controls."""

from __future__ import annotations

from typing import Any

import gradio as gr


REMIX_PRESET_SAME_LANGUAGE = "Same Language"
REMIX_PRESET_TRANSLATION = "Translation"
REMIX_PRESET_CHOICES: tuple[str, ...] = (
    REMIX_PRESET_SAME_LANGUAGE,
    REMIX_PRESET_TRANSLATION,
)
REMIX_PRESET_VALUES: dict[str, tuple[float, float]] = {
    REMIX_PRESET_SAME_LANGUAGE: (0.97, 0.97),
    REMIX_PRESET_TRANSLATION: (0.70, 0.20),
}
REMIX_PRESET_BASE_CLASSES: tuple[str, ...] = (
    "has-info-container",
    "ace-remix-preset-control",
)


def normalize_remix_preset(value: Any) -> str:
    """Return a supported Remix preset name."""

    value_text = str(value or "").strip()
    for choice in REMIX_PRESET_CHOICES:
        if value_text.casefold() == choice.casefold():
            return choice
    return REMIX_PRESET_SAME_LANGUAGE


def remix_preset_values(value: Any) -> tuple[float, float]:
    """Return ``(remix_strength, melody_retention)`` for a Remix preset."""

    return REMIX_PRESET_VALUES[normalize_remix_preset(value)]


def remix_preset_elem_classes(visible: bool) -> list[str]:
    """Return CSS classes that keep the dropdown mounted but mode-hidden."""

    classes = list(REMIX_PRESET_BASE_CLASSES)
    if not visible:
        classes.append("ace-mode-hidden")
    return classes


def apply_remix_preset(value: Any) -> tuple[Any, Any]:
    """Return Gradio updates for Remix Strength and Melody Retention."""

    remix_strength, melody_retention = remix_preset_values(value)
    return gr.update(value=remix_strength), gr.update(value=melody_retention)
