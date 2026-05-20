"""LoRA training sample-generation controls."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.premium_features import (
    DEFAULT_PRESET_CAPTION,
    DEFAULT_PRESET_LYRICS,
)


def build_lora_sample_generation_controls() -> dict[str, object]:
    """Render optional checkpoint sample-generation controls."""

    gr.HTML("<hr><h3>Checkpoint Samples</h3>")
    with gr.Accordion("Generate Samples During Training", open=False):
        with gr.Row():
            lora_sample_enabled = gr.Checkbox(
                label="Generate checkpoint samples",
                value=False,
            )
            lora_sample_every_n_epochs = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=10,
                label="Sample every N epochs",
            )
            lora_sample_seed = gr.Number(
                label="Sample seed",
                value=42,
                precision=0,
            )

        gr.Markdown(
            "Checkpoint samples use the Duration and Inference steps from the "
            "Advanced Generation tab. The sample style prompt and sample lyrics "
            "below are used only for checkpoint samples."
        )

        lora_sample_prompt = gr.Textbox(
            label="Sample style prompt",
            value=DEFAULT_PRESET_CAPTION,
            lines=4,
            max_lines=8,
        )
        lora_sample_lyrics = gr.Textbox(
            label="Sample lyrics",
            value=DEFAULT_PRESET_LYRICS,
            lines=8,
            max_lines=16,
        )
        lora_sample_output_dir = gr.Textbox(
            label="Sample output directory",
            value="./lora_output/samples",
            placeholder="./lora_output/samples",
        )
        with gr.Row():
            lora_sample_offload_training_model = gr.Checkbox(
                label="Offload trainer while sampling",
                value=False,
            )
            lora_sample_offload_generation = gr.Checkbox(
                label="Generation CPU offload",
                value=True,
            )

    return {
        "lora_sample_enabled": lora_sample_enabled,
        "lora_sample_every_n_epochs": lora_sample_every_n_epochs,
        "lora_sample_seed": lora_sample_seed,
        "lora_sample_prompt": lora_sample_prompt,
        "lora_sample_lyrics": lora_sample_lyrics,
        "lora_sample_output_dir": lora_sample_output_dir,
        "lora_sample_offload_training_model": lora_sample_offload_training_model,
        "lora_sample_offload_generation": lora_sample_offload_generation,
    }
