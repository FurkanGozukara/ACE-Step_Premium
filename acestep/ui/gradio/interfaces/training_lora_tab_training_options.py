"""LoRA/DoRA optimizer, scheduler, and timestep controls."""

from __future__ import annotations

import gradio as gr

from acestep.training.optim import (
    OPTIMIZER_CHOICES,
    SCHEDULER_CHOICES,
    optimizer_hyperparameter_defaults,
)


LORA_OPTIMIZER_HYPERPARAMETER_BINDINGS: tuple[tuple[str, str], ...] = (
    ("lora_weight_decay", "weight_decay"),
    ("lora_adam_beta1", "adam_beta1"),
    ("lora_adam_beta2", "adam_beta2"),
    ("lora_adam_epsilon", "adam_epsilon"),
    ("lora_adamw8bit_min_8bit_size", "adamw8bit_min_8bit_size"),
    ("lora_adamw8bit_percentile_clipping", "adamw8bit_percentile_clipping"),
    ("lora_adamw8bit_block_wise", "adamw8bit_block_wise"),
    ("lora_adamw8bit_paged", "adamw8bit_paged"),
    ("lora_adafactor_epsilon1", "adafactor_epsilon1"),
    ("lora_adafactor_epsilon2", "adafactor_epsilon2"),
    ("lora_adafactor_clip_threshold", "adafactor_clip_threshold"),
    ("lora_adafactor_decay_rate", "adafactor_decay_rate"),
    ("lora_adafactor_beta1", "adafactor_beta1"),
    ("lora_adafactor_scale_parameter", "adafactor_scale_parameter"),
    ("lora_adafactor_relative_step", "adafactor_relative_step"),
    ("lora_adafactor_warmup_init", "adafactor_warmup_init"),
)
LORA_OPTIMIZER_HYPERPARAMETER_KEYS = tuple(
    ui_key for ui_key, _param_key in LORA_OPTIMIZER_HYPERPARAMETER_BINDINGS
)
LORA_OPTIMIZER_PARAMETER_ROW_KEYS = (
    "lora_adam_parameters_row",
    "lora_adamw8bit_parameters_row",
    "lora_adafactor_parameters_row",
)
LORA_OPTIMIZER_UI_UPDATE_KEYS = (
    *LORA_OPTIMIZER_PARAMETER_ROW_KEYS,
    *LORA_OPTIMIZER_HYPERPARAMETER_KEYS,
)
_LORA_ADAM_HYPERPARAMETER_KEYS = frozenset(
    {
        "lora_adam_beta1",
        "lora_adam_beta2",
        "lora_adam_epsilon",
    }
)
_LORA_ADAMW8BIT_HYPERPARAMETER_KEYS = frozenset(
    {
        "lora_adamw8bit_min_8bit_size",
        "lora_adamw8bit_percentile_clipping",
        "lora_adamw8bit_block_wise",
        "lora_adamw8bit_paged",
    }
)
_LORA_ADAFACTOR_HYPERPARAMETER_KEYS = frozenset(
    {
        "lora_adafactor_epsilon1",
        "lora_adafactor_epsilon2",
        "lora_adafactor_clip_threshold",
        "lora_adafactor_decay_rate",
        "lora_adafactor_beta1",
        "lora_adafactor_scale_parameter",
        "lora_adafactor_relative_step",
        "lora_adafactor_warmup_init",
    }
)


def _selected_optimizer(optimizer_type: object) -> str:
    selected = str(optimizer_type or "adamw").strip().casefold()
    return selected if selected in OPTIMIZER_CHOICES else "adamw"


def lora_optimizer_parameter_row_updates(optimizer_type: object):
    """Return Gradio updates for optimizer-specific parameter rows."""

    selected = _selected_optimizer(optimizer_type)
    return (
        gr.update(visible=selected in {"adamw", "adamw8bit"}),
        gr.update(visible=selected == "adamw8bit"),
        gr.update(visible=selected == "adafactor"),
    )


def _optimizer_hyperparameter_visible(ui_key: str, selected: str) -> bool:
    if ui_key == "lora_weight_decay":
        return True
    if ui_key in _LORA_ADAM_HYPERPARAMETER_KEYS:
        return True
    if ui_key in _LORA_ADAMW8BIT_HYPERPARAMETER_KEYS:
        return True
    if ui_key in _LORA_ADAFACTOR_HYPERPARAMETER_KEYS:
        return selected == "adafactor"
    return True


def lora_optimizer_hyperparameter_updates(optimizer_type: object):
    """Return Gradio updates for optimizer hyperparameters after dropdown changes."""

    selected = _selected_optimizer(optimizer_type)
    defaults = optimizer_hyperparameter_defaults(selected)
    return tuple(
        gr.update(
            value=defaults[param_key],
            visible=_optimizer_hyperparameter_visible(ui_key, selected),
        )
        for ui_key, param_key in LORA_OPTIMIZER_HYPERPARAMETER_BINDINGS
    )


def lora_optimizer_ui_updates(optimizer_type: object):
    """Return all optimizer UI updates after dropdown changes."""

    return (
        *lora_optimizer_parameter_row_updates(optimizer_type),
        *lora_optimizer_hyperparameter_updates(optimizer_type),
    )


def build_lora_training_option_controls(
    default_optimizer: object,
    default_scheduler: object = "constant",
) -> dict[str, object]:
    """Render advanced LoRA/DoRA training option controls."""

    optimizer_defaults = optimizer_hyperparameter_defaults(
        str(default_optimizer or "adamw8bit")
    )
    selected_optimizer = _selected_optimizer(default_optimizer)

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

    with gr.Row():
        lora_weight_decay = gr.Number(
            label="Weight decay",
            value=optimizer_defaults["weight_decay"],
            info=(
                "Decoupled weight decay passed to the selected optimizer. "
                "AdamW defaults to 0.01; Adafactor defaults to 0.0."
            ),
            elem_classes=["has-info-container"],
        )

    with gr.Row() as lora_adam_parameters_row:
        lora_adam_beta1 = gr.Number(
            label="Adam beta1",
            value=optimizer_defaults["adam_beta1"],
            visible=selected_optimizer in {"adamw", "adamw8bit"},
            info="First-moment decay for AdamW and AdamW8bit.",
            elem_classes=["has-info-container"],
        )
        lora_adam_beta2 = gr.Number(
            label="Adam beta2",
            value=optimizer_defaults["adam_beta2"],
            visible=selected_optimizer in {"adamw", "adamw8bit"},
            info="Second-moment decay for AdamW and AdamW8bit.",
            elem_classes=["has-info-container"],
        )
        lora_adam_epsilon = gr.Number(
            label="Adam epsilon",
            value=optimizer_defaults["adam_epsilon"],
            visible=selected_optimizer in {"adamw", "adamw8bit"},
            info="Small denominator stabilizer for AdamW and AdamW8bit.",
            elem_classes=["has-info-container"],
        )

    with gr.Row(visible=True) as lora_adamw8bit_parameters_row:
        lora_adamw8bit_min_8bit_size = gr.Number(
            label="8-bit min size",
            value=optimizer_defaults["adamw8bit_min_8bit_size"],
            precision=0,
            info="Minimum tensor size that bitsandbytes stores in 8-bit state.",
            elem_classes=["has-info-container"],
        )
        lora_adamw8bit_percentile_clipping = gr.Number(
            label="8-bit percentile clipping",
            value=optimizer_defaults["adamw8bit_percentile_clipping"],
            precision=0,
            info="bitsandbytes gradient-norm percentile clipping. 100 disables clipping.",
            elem_classes=["has-info-container"],
        )
        lora_adamw8bit_block_wise = gr.Checkbox(
            label="8-bit block-wise",
            value=optimizer_defaults["adamw8bit_block_wise"],
            info="Quantize optimizer state independently per block.",
            elem_classes=["has-info-container"],
        )
        lora_adamw8bit_paged = gr.Checkbox(
            label="8-bit paged state",
            value=optimizer_defaults["adamw8bit_paged"],
            info="Allow bitsandbytes paged optimizer state via CUDA unified memory.",
            elem_classes=["has-info-container"],
        )

    with gr.Row(visible=True) as lora_adafactor_parameters_row:
        lora_adafactor_epsilon1 = gr.Number(
            label="Adafactor eps1",
            value=optimizer_defaults["adafactor_epsilon1"],
            visible=selected_optimizer == "adafactor",
            info="Adafactor squared-gradient stabilizer.",
            elem_classes=["has-info-container"],
        )
        lora_adafactor_epsilon2 = gr.Number(
            label="Adafactor eps2",
            value=optimizer_defaults["adafactor_epsilon2"],
            visible=selected_optimizer == "adafactor",
            info="Adafactor parameter-scale stabilizer.",
            elem_classes=["has-info-container"],
        )
        lora_adafactor_clip_threshold = gr.Number(
            label="Adafactor clip",
            value=optimizer_defaults["adafactor_clip_threshold"],
            visible=selected_optimizer == "adafactor",
            info="RMS update clipping threshold for Adafactor.",
            elem_classes=["has-info-container"],
        )
        lora_adafactor_decay_rate = gr.Number(
            label="Adafactor decay",
            value=optimizer_defaults["adafactor_decay_rate"],
            visible=selected_optimizer == "adafactor",
            info="Adafactor second-moment decay rate.",
            elem_classes=["has-info-container"],
        )
        lora_adafactor_beta1 = gr.Number(
            label="Adafactor beta1",
            value=optimizer_defaults["adafactor_beta1"],
            visible=selected_optimizer == "adafactor",
            info="First-moment decay for Adafactor. 0 disables beta1.",
            elem_classes=["has-info-container"],
        )
        lora_adafactor_scale_parameter = gr.Checkbox(
            label="Adafactor scale parameter",
            value=optimizer_defaults["adafactor_scale_parameter"],
            visible=selected_optimizer == "adafactor",
            info="Use Adafactor parameter-scale learning-rate adjustment.",
            elem_classes=["has-info-container"],
        )
        lora_adafactor_relative_step = gr.Checkbox(
            label="Adafactor relative step",
            value=optimizer_defaults["adafactor_relative_step"],
            visible=selected_optimizer == "adafactor",
            info=(
                "Use Adafactor internal relative-step LR. This ignores the "
                "manual learning rate inside the optimizer."
            ),
            elem_classes=["has-info-container"],
        )
        lora_adafactor_warmup_init = gr.Checkbox(
            label="Adafactor warmup init",
            value=optimizer_defaults["adafactor_warmup_init"],
            visible=selected_optimizer == "adafactor",
            info="Warm up Adafactor's internal relative-step LR.",
            elem_classes=["has-info-container"],
        )

    return {
        "lora_optimizer_type": lora_optimizer_type,
        "lora_adam_parameters_row": lora_adam_parameters_row,
        "lora_adamw8bit_parameters_row": lora_adamw8bit_parameters_row,
        "lora_adafactor_parameters_row": lora_adafactor_parameters_row,
        "lora_weight_decay": lora_weight_decay,
        "lora_adam_beta1": lora_adam_beta1,
        "lora_adam_beta2": lora_adam_beta2,
        "lora_adam_epsilon": lora_adam_epsilon,
        "lora_adamw8bit_min_8bit_size": lora_adamw8bit_min_8bit_size,
        "lora_adamw8bit_percentile_clipping": lora_adamw8bit_percentile_clipping,
        "lora_adamw8bit_block_wise": lora_adamw8bit_block_wise,
        "lora_adamw8bit_paged": lora_adamw8bit_paged,
        "lora_adafactor_epsilon1": lora_adafactor_epsilon1,
        "lora_adafactor_epsilon2": lora_adafactor_epsilon2,
        "lora_adafactor_clip_threshold": lora_adafactor_clip_threshold,
        "lora_adafactor_decay_rate": lora_adafactor_decay_rate,
        "lora_adafactor_beta1": lora_adafactor_beta1,
        "lora_adafactor_scale_parameter": lora_adafactor_scale_parameter,
        "lora_adafactor_relative_step": lora_adafactor_relative_step,
        "lora_adafactor_warmup_init": lora_adafactor_warmup_init,
        "lora_scheduler_type": lora_scheduler_type,
        "lora_timestep_mode": lora_timestep_mode,
        "lora_adaptive_timestep_ratio": lora_adaptive_timestep_ratio,
        "lora_validation_split_percent": lora_validation_split_percent,
        "lora_validation_split_info": lora_validation_split_info,
    }
