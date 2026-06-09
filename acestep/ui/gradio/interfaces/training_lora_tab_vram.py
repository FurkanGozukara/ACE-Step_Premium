"""LoRA training VRAM optimization controls."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.interfaces.training_lora_vram_presets import (
    default_lora_vram_control_values,
)


def build_lora_vram_controls() -> dict[str, object]:
    """Render memory optimization controls for LoRA training."""

    lora_vram_defaults = default_lora_vram_control_values()

    with gr.Accordion("LoRA VRAM Optimizations", open=True):
        with gr.Row():
            lora_gradient_checkpointing = gr.Checkbox(
                label="Gradient checkpointing",
                value=lora_vram_defaults["gradient_checkpointing"],
            )
            lora_activation_cpu_offload = gr.Checkbox(
                label="Activation CPU offload",
                value=lora_vram_defaults["activation_cpu_offload"],
            )
            lora_offload_non_decoder = gr.Checkbox(
                label="Offload unused encoders",
                value=lora_vram_defaults["offload_non_decoder"],
            )

        with gr.Row():
            lora_keep_frozen_bf16 = gr.Checkbox(
                label="Keep frozen base in bf16/fp16",
                value=lora_vram_defaults["keep_frozen_base_in_compute_dtype"],
            )
            lora_compile_model = gr.Checkbox(
                label="Compile Model",
                value=lora_vram_defaults["compile_model"],
                info="Use torch.compile on the training decoder. First step may be slower.",
            )
            lora_base_quantization = gr.Dropdown(
                label="Frozen base quantization",
                choices=["Disabled", "FP8 scaled"],
                value=lora_vram_defaults["base_quantization"],
            )

        lora_empty_cache_every_n_steps = gr.Slider(
            minimum=0,
            maximum=100,
            step=1,
            value=lora_vram_defaults["empty_cache_every_n_steps"],
            label="Empty CUDA cache every N optimizer steps",
        )

    return {
        "lora_gradient_checkpointing": lora_gradient_checkpointing,
        "lora_activation_cpu_offload": lora_activation_cpu_offload,
        "lora_offload_non_decoder": lora_offload_non_decoder,
        "lora_keep_frozen_bf16": lora_keep_frozen_bf16,
        "lora_compile_model": lora_compile_model,
        "lora_base_quantization": lora_base_quantization,
        "lora_empty_cache_every_n_steps": lora_empty_cache_every_n_steps,
    }
