"""LoRA training VRAM optimization controls."""

from __future__ import annotations

import gradio as gr


def build_lora_vram_controls() -> dict[str, object]:
    """Render memory optimization controls for LoRA training."""

    with gr.Accordion("LoRA VRAM Optimizations", open=True):
        with gr.Row():
            lora_gradient_checkpointing = gr.Checkbox(
                label="Gradient checkpointing",
                value=True,
            )
            lora_activation_cpu_offload = gr.Checkbox(
                label="Activation CPU offload",
                value=False,
            )
            lora_offload_non_decoder = gr.Checkbox(
                label="Offload unused encoders",
                value=True,
            )

        with gr.Row():
            lora_keep_frozen_bf16 = gr.Checkbox(
                label="Keep frozen base in bf16/fp16",
                value=True,
            )
            lora_use_8bit_adam = gr.Checkbox(
                label="8-bit Adam optimizer",
                value=True,
            )
            lora_base_quantization = gr.Dropdown(
                label="Frozen base quantization",
                choices=["Disabled", "FP8 scaled"],
                value="Disabled",
            )

        lora_empty_cache_every_n_steps = gr.Slider(
            minimum=0,
            maximum=100,
            step=1,
            value=10,
            label="Empty CUDA cache every N optimizer steps",
        )

    return {
        "lora_gradient_checkpointing": lora_gradient_checkpointing,
        "lora_activation_cpu_offload": lora_activation_cpu_offload,
        "lora_offload_non_decoder": lora_offload_non_decoder,
        "lora_keep_frozen_bf16": lora_keep_frozen_bf16,
        "lora_use_8bit_adam": lora_use_8bit_adam,
        "lora_base_quantization": lora_base_quantization,
        "lora_empty_cache_every_n_steps": lora_empty_cache_every_n_steps,
    }
