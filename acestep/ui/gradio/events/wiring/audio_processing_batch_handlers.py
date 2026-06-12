"""Batch-folder Audio Processing Gradio handlers."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.batch import run_batch_audio_processing
from acestep.audio_processing.settings import settings_from_ui_values


def process_batch_folder(
    input_folder: str,
    output_folder: str,
    recursive: bool,
    *settings_values: Any,
):
    """Stream batch-folder processing status and generated files."""

    settings = settings_from_ui_values(settings_values)
    for status, files in run_batch_audio_processing(
        input_folder,
        output_folder,
        bool(recursive),
        settings,
    ):
        yield status, gr.update(value=files, visible=bool(files))
