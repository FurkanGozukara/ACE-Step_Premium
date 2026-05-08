"""Dataset tools page for the premium ACE-Step Gradio shell."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.interfaces.dataset import create_dataset_section


def create_dataset_page(dataset_handler) -> dict[str, gr.components.Component]:
    """Build the dedicated dataset explorer page."""

    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Dataset Tools</span>
              <h2>Inspect dataset items, preview audio, and push samples into generation.</h2>
              <p>
                The dataset explorer is now a normal page instead of a hidden accordion at the top of the app.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            dataset_section = create_dataset_section(
                dataset_handler,
                title="Dataset Explorer",
                open=True,
                visible=True,
            )

    return dataset_section
