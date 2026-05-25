"""Custom trigger controls for the training dataset tab."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.i18n import t


def build_custom_trigger_controls() -> dict[str, object]:
    """Render custom trigger controls and return component handles."""

    with gr.Row():
        with gr.Column(scale=2):
            custom_tag = gr.Textbox(
                label=t("training.custom_tag"),
                placeholder="e.g., 8bit_retro, my_style",
                info=t("training.custom_tag_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Column(scale=1):
            use_only_custom_trigger = gr.Checkbox(
                label=t("training.use_only_custom_trigger"),
                value=False,
                info=t("training.use_only_custom_trigger_info"),
                elem_classes=["has-info-container"],
            )

    return {
        "custom_tag": custom_tag,
        "use_only_custom_trigger": use_only_custom_trigger,
    }
