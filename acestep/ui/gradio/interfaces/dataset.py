"""Dataset explorer section component definitions for the Gradio UI."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.interfaces.dataset_ngram_browser import (
    build_dataset_import_controls,
    build_ngram_browser_controls,
)
from acestep.ui.gradio.interfaces.dataset_preview_panel import (
    build_dataset_preview_controls,
)


def create_dataset_section(
    dataset_handler,
    *,
    title: str = "Dataset Explorer",
    open: bool = True,
    visible: bool = False,
) -> dict:
    """Create the dataset explorer section."""

    del dataset_handler

    with gr.Accordion(title, open=open, visible=visible):
        section: dict[str, object] = {}
        section.update(build_dataset_import_controls())
        section.update(build_ngram_browser_controls())
        section.update(build_dataset_preview_controls())
    return section
