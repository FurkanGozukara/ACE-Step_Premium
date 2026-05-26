"""LoRA training output and resume path controls."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.i18n import t


def build_lora_path_controls(default_output_dir: str) -> dict[str, object]:
    """Render LoRA output-directory and resume-state controls."""

    with gr.Row():
        with gr.Column(scale=3):
            lora_output_dir = gr.Textbox(
                label=t("training.output_dir"),
                value=default_output_dir,
                placeholder=default_output_dir,
                info=t("training.output_dir_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Column(scale=1):
            lora_output_dir_browse_btn = gr.Button(
                "Browse Output Folder",
                variant="secondary",
            )
            lora_open_output_dir_btn = gr.Button(
                "Open Output Folder",
                variant="secondary",
            )

    with gr.Row():
        with gr.Column(scale=3):
            resume_checkpoint_dir = gr.Textbox(
                label="Resume Training State",
                placeholder="./Loras/my_lora/epoch-200-training_resume_state.pt",
                info=(
                    "Path to a saved LoRA training_resume_state .pt file from "
                    "<Output Directory>/<LoRA Training Name>"
                ),
                elem_classes=["has-info-container"],
            )
        with gr.Column(scale=1):
            resume_checkpoint_dir_browse_btn = gr.Button(
                "Browse Training State",
                variant="secondary",
            )

    return {
        "lora_output_dir": lora_output_dir,
        "lora_output_dir_browse_btn": lora_output_dir_browse_btn,
        "lora_open_output_dir_btn": lora_open_output_dir_btn,
        "resume_checkpoint_dir": resume_checkpoint_dir,
        "resume_checkpoint_dir_browse_btn": resume_checkpoint_dir_browse_btn,
    }
