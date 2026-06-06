"""Stage controls for the Audio Processing page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.presets import DEFAULT_STAGE_VALUES, STAGE_KEYS, STAGE_LABELS


def add_stage_controls(controls: dict[str, Any]) -> None:
    """Add Audio Enhancement and Pre-Mastering stage controls."""

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Audio Enhancement")
            controls["ap_toggle_audio_enhancement_btn"] = gr.Button(
                "Check / Uncheck All",
                variant="secondary",
                size="sm",
            )
            for key in STAGE_KEYS[:6]:
                _add_stage_control(controls, key)
        with gr.Column(scale=1):
            gr.Markdown("### Pre-Mastering")
            for key in STAGE_KEYS[6:]:
                _add_stage_control(controls, key)


def _add_stage_control(controls: dict[str, Any], key: str) -> None:
    """Add one enabled checkbox and value slider to the page controls."""

    minimum, maximum, step = _slider_bounds(key)
    with gr.Row(equal_height=True):
        controls[f"ap_{key}_enabled"] = gr.Checkbox(
            label=STAGE_LABELS[key],
            value=True,
            scale=1,
        )
        controls[f"ap_{key}"] = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            step=step,
            value=DEFAULT_STAGE_VALUES[key],
            label=STAGE_LABELS[key],
            scale=4,
        )


def _slider_bounds(key: str) -> tuple[float, float, float]:
    """Return slider bounds for an audio-processing stage."""

    if key == "stereo":
        return 1.0, 2.0, 0.05
    if key == "lufs":
        return -24.0, -8.0, 0.5
    return 0.0, 1.0, 0.05
