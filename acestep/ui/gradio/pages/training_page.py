"""Training page wrapper for the premium ACE-Step Gradio shell."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.interfaces.training import create_training_section


def create_training_page(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dedicated training page."""

    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Training</span>
              <h2>Build datasets and train LoRA or LoKr adapters without crowding the generation page.</h2>
              <p>
                Dataset builder, preprocessing, and training controls stay grouped here so the Create page remains focused on inference.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            training_section = create_training_section(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                init_params=init_params,
            )

    return training_section
