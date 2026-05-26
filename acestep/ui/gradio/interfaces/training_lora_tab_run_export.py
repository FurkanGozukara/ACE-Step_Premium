"""LoRA tab run and export controls."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from acestep.ui.gradio.events.training.schedule_defaults import (
    training_schedule_defaults_for_model,
)
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.interfaces.training_lora_tab_paths import (
    build_lora_path_controls,
)
from acestep.ui.gradio.interfaces.training_lora_tab_training_options import (
    build_lora_training_option_controls,
)
from acestep.ui.gradio.interfaces.training_lora_vram_presets import (
    default_lora_vram_control_values,
)


def default_lora_output_dir() -> str:
    """Return the default LoRA output directory under the installation root."""

    install_root = Path(__file__).resolve().parents[4]
    return str(install_root / "Loras")


def build_lora_run_and_export_controls(
    *,
    epoch_min: int,
    epoch_step: int,
    epoch_default: int,
    model_config: str | None = None,
) -> dict[str, object]:
    """Render LoRA training-run controls for the training tab."""

    default_output_dir = default_lora_output_dir()
    schedule_defaults = training_schedule_defaults_for_model(model_config)
    lora_vram_defaults = default_lora_vram_control_values()
    gr.HTML(f"<hr><h3>🎛️ {t('training.train_section_params')}</h3>")

    with gr.Row():
        learning_rate = gr.Number(
            label=t("training.learning_rate"),
            value=3e-4,
            info=t("training.learning_rate_info"),
            elem_classes=["has-info-container"],
        )

        train_epochs = gr.Slider(
            minimum=epoch_min,
            maximum=4000,
            step=epoch_step,
            value=epoch_default,
            label=t("training.max_epochs"),
            info=t("training.max_epochs_info"),
            elem_classes=["has-info-container"],
        )

        train_batch_size = gr.Slider(
            minimum=1,
            maximum=8,
            step=1,
            value=1,
            label=t("training.batch_size"),
            info=t("training.batch_size_info"),
            elem_classes=["has-info-container"],
        )

        gradient_accumulation = gr.Slider(
            minimum=1,
            maximum=16,
            step=1,
            value=1,
            label=t("training.gradient_accumulation"),
            info=t("training.gradient_accumulation_info"),
            elem_classes=["has-info-container"],
        )

    training_step_estimate = gr.Markdown(
        "Load a tensor dataset to calculate total training steps."
    )

    with gr.Row():
        save_every_n_epochs = gr.Slider(
            minimum=0,
            maximum=1000,
            step=1,
            value=10,
            label=t("training.save_every_n_epochs"),
            info=t("training.save_every_n_epochs_info"),
            elem_classes=["has-info-container"],
        )

        lora_save_best = gr.Checkbox(
            label="Save best",
            value=True,
            info=(
                "Keeps one best checkpoint in <output>/best. A lower validation "
                "loss replaces it; if no validation split exists, epoch training "
                "loss is used."
            ),
            elem_classes=["has-info-container"],
        )

        training_shift = gr.Slider(
            minimum=schedule_defaults["shift_minimum"],
            maximum=schedule_defaults["shift_maximum"],
            step=schedule_defaults["shift_step"],
            value=schedule_defaults["shift"],
            label=t("training.shift"),
            info=t("training.shift_info"),
            elem_classes=["has-info-container"],
        )

        training_num_inference_steps = gr.Slider(
            minimum=schedule_defaults["num_inference_steps_minimum"],
            maximum=schedule_defaults["num_inference_steps_maximum"],
            step=1,
            value=schedule_defaults["num_inference_steps"],
            label=t("training.num_inference_steps"),
            info=t("training.num_inference_steps_info"),
            elem_classes=["has-info-container"],
        )

        training_seed = gr.Number(
            label=t("training.seed"),
            value=42,
            precision=0,
        )

    option_controls = build_lora_training_option_controls(
        lora_vram_defaults.get("optimizer_type", "adamw8bit")
    )
    path_controls = build_lora_path_controls(default_output_dir)

    gr.HTML("<hr>")

    with gr.Row():
        with gr.Column(scale=1):
            training_subprocess = gr.Checkbox(
                label="Run in isolated subprocess",
                value=True,
                info="Frees VRAM/RAM when the training worker exits.",
                elem_classes=["has-info-container"],
            )
            start_training_btn = gr.Button(
                t("training.start_training_btn"),
                variant="primary",
                size="lg",
            )
        with gr.Column(scale=1):
            stop_training_btn = gr.Button(
                t("training.stop_training_btn"),
                variant="stop",
                size="lg",
            )

    training_progress = gr.Textbox(
        label=t("training.training_progress"),
        interactive=False,
        lines=2,
    )

    with gr.Row():
        training_log = gr.Textbox(
            label=t("training.training_log"),
            interactive=False,
            lines=10,
            max_lines=15,
            scale=1,
        )
        training_loss_plot = gr.Plot(
            label=t("training.training_loss_title"),
            scale=1,
        )

    return {
        "learning_rate": learning_rate,
        "train_epochs": train_epochs,
        "train_batch_size": train_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "training_step_estimate": training_step_estimate,
        "save_every_n_epochs": save_every_n_epochs,
        "lora_save_best": lora_save_best,
        "training_shift": training_shift,
        "training_num_inference_steps": training_num_inference_steps,
        "training_seed": training_seed,
        **option_controls,
        **path_controls,
        "training_subprocess": training_subprocess,
        "start_training_btn": start_training_btn,
        "stop_training_btn": stop_training_btn,
        "training_progress": training_progress,
        "training_log": training_log,
        "training_loss_plot": training_loss_plot,
    }
