"""SAM Audio Segment Gradio page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from .sam_audio_page_io import add_batch_controls, add_single_file_controls
from .sam_audio_page_settings import add_generation_controls, add_prompt_runtime_controls


def create_sam_audio_page() -> dict[str, Any]:
    """Build the SAM Audio Segment tab controls."""

    controls: dict[str, Any] = {}
    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        add_single_file_controls(controls)
        add_generation_controls(controls)
        add_prompt_runtime_controls(controls)
        add_batch_controls(controls)
    return controls
