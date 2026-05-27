"""LoRA/DoRA optimizer, scheduler, and timestep controls."""

from __future__ import annotations

import gradio as gr

from acestep.training.optim import OPTIMIZER_CHOICES, SCHEDULER_CHOICES


def build_lora_training_option_controls(
    default_optimizer: object,
    default_scheduler: object = "constant",
) -> dict[str, object]:
    """Render advanced LoRA/DoRA training option controls."""

    with gr.Row():
        lora_optimizer_type = gr.Dropdown(
            label="Optimizer",
            choices=list(OPTIMIZER_CHOICES),
            value=str(default_optimizer or "adamw8bit"),
            info=(
                "Optimizer used for trainable LoRA/DoRA parameters. adamw8bit "
                "saves VRAM on CUDA, adamw is the standard full-state optimizer, "
                "and adafactor uses less optimizer state."
            ),
            elem_classes=["has-info-container"],
        )
        lora_scheduler_type = gr.Dropdown(
            label="Scheduler",
            choices=list(SCHEDULER_CHOICES),
            value=str(default_scheduler or "constant"),
            info=(
                "Learning-rate schedule. Constant keeps LR flat and is the "
                "default; cosine/linear decay LR over the run; restarts "
                "periodically raises LR again."
            ),
            elem_classes=["has-info-container"],
        )
        lora_timestep_mode = gr.Dropdown(
            label="Timestep mode",
            choices=["continuous", "discrete"],
            value="continuous",
            info=(
                "continuous samples logit-normal timesteps like ACE training and "
                "is the recommended default. discrete keeps the older shifted "
                "inference-step sampler for compatibility."
            ),
            elem_classes=["has-info-container"],
        )
        lora_adaptive_timestep_ratio = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.0,
            label="Adaptive timestep",
            info=(
                "Fraction of each batch sampled from loss-weighted timestep bins. "
                "0.0 disables adaptive sampling and is the safest default."
            ),
            elem_classes=["has-info-container"],
        )
        with gr.Column():
            lora_validation_split_percent = gr.Slider(
                minimum=0,
                maximum=99,
                step=1,
                value=0,
                label="Validation split %",
                info=(
                    "Percent of loaded tensor samples held out for validation. "
                    "0 disables validation; any non-zero split keeps at least "
                    "one sample for training."
                ),
                elem_classes=["has-info-container"],
            )
            lora_validation_split_info = gr.Markdown(
                "Load a tensor dataset to preview the validation split."
            )

    return {
        "lora_optimizer_type": lora_optimizer_type,
        "lora_scheduler_type": lora_scheduler_type,
        "lora_timestep_mode": lora_timestep_mode,
        "lora_adaptive_timestep_ratio": lora_adaptive_timestep_ratio,
        "lora_validation_split_percent": lora_validation_split_percent,
        "lora_validation_split_info": lora_validation_split_info,
    }
