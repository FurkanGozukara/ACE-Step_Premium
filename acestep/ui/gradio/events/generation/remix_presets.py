"""Remix preset helpers for paired strength controls."""

from __future__ import annotations

from typing import Any

import gradio as gr


REMIX_PRESET_DEFAULT = "Default"
REMIX_PRESET_SAME_LYRICS_BIG_CHANGE = "Same Lyrics Big Change"
REMIX_PRESET_SAME_LYRICS_MEDIUM_CHANGE = "Same Lyrics Medium Change"
REMIX_PRESET_DIFFERENT_LYRICS = "Different Lyrics"
REMIX_PRESET_SAME_LANGUAGE = "Same Language"
REMIX_PRESET_TRANSLATION = "Translation"
REMIX_PRESET_CHOICES: tuple[str, ...] = (
    REMIX_PRESET_DEFAULT,
    REMIX_PRESET_SAME_LYRICS_BIG_CHANGE,
    REMIX_PRESET_SAME_LYRICS_MEDIUM_CHANGE,
    REMIX_PRESET_DIFFERENT_LYRICS,
)
REMIX_PRESET_VALUES: dict[str, tuple[float, float]] = {
    REMIX_PRESET_DEFAULT: (0.50, 0.50),
    REMIX_PRESET_SAME_LYRICS_BIG_CHANGE: (0.10, 0.10),
    REMIX_PRESET_SAME_LYRICS_MEDIUM_CHANGE: (0.25, 0.25),
    REMIX_PRESET_DIFFERENT_LYRICS: (0.70, 0.15),
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
    return REMIX_PRESET_DEFAULT


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
