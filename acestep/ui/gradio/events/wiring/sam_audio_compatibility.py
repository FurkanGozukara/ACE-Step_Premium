"""UI compatibility updates for SAM-Audio settings."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.sam_audio_segment.settings import SAM_AUDIO_LONG_MODE_MULTIDIFFUSION

PROMPT_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Text", "text"),
    ("Span", "span"),
    ("Visual mask", "visual"),
)
TEXT_SPAN_PROMPT_CHOICES: tuple[tuple[str, str], ...] = (
    ("Text", "text"),
    ("Span", "span"),
)
TEXT_ONLY_PROMPT_CHOICES: tuple[tuple[str, str], ...] = (("Text", "text"),)


def apply_sam_audio_compatibility(
    prompt_mode: str | None,
    low_vram_lite: bool,
    long_audio_mode: str | None,
) -> tuple[Any, ...]:
    """Return UI updates that disable invalid SAM-Audio setting combinations."""

    lite_enabled = bool(low_vram_lite)
    multidiffusion_enabled = long_audio_mode == SAM_AUDIO_LONG_MODE_MULTIDIFFUSION
    prompt_value = _compatible_prompt_mode(
        prompt_mode,
        lite_enabled=lite_enabled,
        multidiffusion_enabled=multidiffusion_enabled,
    )
    text_only = lite_enabled or multidiffusion_enabled
    rankers_available = not text_only
    spans_available = prompt_value == "text" and not text_only
    anchors_available = not multidiffusion_enabled
    visual_available = (
        prompt_value == "visual"
        and not lite_enabled
        and not multidiffusion_enabled
    )
    return (
        gr.update(
            choices=_prompt_choices(lite_enabled, multidiffusion_enabled),
            value=prompt_value,
            interactive=not multidiffusion_enabled,
        ),
        gr.update(value="none", interactive=False)
        if not rankers_available
        else gr.update(interactive=True),
        gr.update(value=False, interactive=False)
        if not spans_available
        else gr.update(interactive=True),
        gr.update(value=1, interactive=False)
        if multidiffusion_enabled
        else gr.update(interactive=True),
        gr.update(value=False, interactive=False)
        if multidiffusion_enabled
        else gr.update(interactive=True),
        gr.update(value="", interactive=False)
        if multidiffusion_enabled
        else gr.update(interactive=True),
        gr.update(interactive=anchors_available),
        gr.update(interactive=anchors_available),
        gr.update(interactive=anchors_available),
        gr.update(value=None, interactive=visual_available),
    )


def _compatible_prompt_mode(
    value: str | None,
    *,
    lite_enabled: bool,
    multidiffusion_enabled: bool,
) -> str:
    """Return a prompt mode supported by the selected runtime options."""

    normalized = str(value or "text").strip()
    if multidiffusion_enabled:
        return "text"
    if lite_enabled and normalized == "visual":
        return "text"
    return normalized if normalized in {"text", "span", "visual"} else "text"


def _prompt_choices(
    lite_enabled: bool,
    multidiffusion_enabled: bool,
) -> tuple[tuple[str, str], ...]:
    """Return prompt-mode choices for the selected runtime options."""

    if multidiffusion_enabled:
        return TEXT_ONLY_PROMPT_CHOICES
    if lite_enabled:
        return TEXT_SPAN_PROMPT_CHOICES
    return PROMPT_MODE_CHOICES
