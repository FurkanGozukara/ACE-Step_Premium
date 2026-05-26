"""LoRA tab dataset and adapter-setting controls."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.interfaces.training_dataset_model_selector import (
    build_lora_training_model_selector,
)
from acestep.ui.gradio.interfaces.training_lora_vram_presets import (
    build_lora_vram_preset_dropdown,
    default_lora_vram_control_values,
)


def build_lora_dataset_and_adapter_controls(
    dit_handler=None,
    init_params: dict | None = None,
) -> dict[str, object]:
    """Render LoRA dataset selector and adapter-parameter controls."""

    lora_vram_defaults = default_lora_vram_control_values()

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(f"<h3>📊 {t('training.train_section_tensors')}</h3>")
            gr.Markdown(t("training.train_tensor_selection_desc"))

            training_tensor_dir = gr.Textbox(
                label=t("training.preprocessed_tensors_dir"),
                placeholder="./datasets/preprocessed_tensors",
                value="./datasets/preprocessed_tensors",
                info=t("training.preprocessed_tensors_info"),
                elem_classes=["has-info-container"],
            )

            training_tensor_dir_browse_btn = gr.Button(
                "Browse Tensor Folder",
                variant="secondary",
            )
            load_dataset_btn = gr.Button(t("training.load_dataset_btn"), variant="secondary")

            training_dataset_info = gr.Textbox(
                label=t("training.dataset_info"),
                interactive=False,
                lines=3,
            )

            with gr.Row():
                with gr.Column(scale=2):
                    model_controls = build_lora_training_model_selector(
                        dit_handler=dit_handler,
                        init_params=init_params,
                    )
                with gr.Column(scale=1):
                    lora_vram_preset = build_lora_vram_preset_dropdown()

        with gr.Column(scale=1):
            gr.HTML(f"<h3>⚙️ {t('training.train_section_lora')}</h3>")

            lora_adapter_type = gr.Dropdown(
                label="Adapter type",
                choices=["lora", "dora"],
                value="lora",
                info=(
                    "LoRA is the standard low-rank adapter. DoRA uses the same "
                    "rank/alpha/dropout controls plus PEFT weight decomposition "
                    "for magnitude and direction learning."
                ),
                elem_classes=["has-info-container"],
            )

            lora_adapter_info = gr.Markdown(
                "LoRA selected: standard PEFT LoRA training."
            )

            lora_name = gr.Textbox(
                label=t("training.lora_name"),
                placeholder="my-awesome-song",
                info=t("training.lora_name_info"),
                elem_classes=["has-info-container"],
            )

            with gr.Row(equal_height=True):
                lora_rank = gr.Slider(
                    minimum=4,
                    maximum=256,
                    step=4,
                    value=lora_vram_defaults["lora_rank"],
                    label=t("training.lora_rank"),
                    info=t("training.lora_rank_info"),
                    elem_classes=["has-info-container"],
                )

                lora_alpha = gr.Slider(
                    minimum=4,
                    maximum=512,
                    step=4,
                    value=lora_vram_defaults["lora_alpha"],
                    label=t("training.lora_alpha"),
                    info=t("training.lora_alpha_info"),
                    elem_classes=["has-info-container"],
                )

            lora_dropout = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                step=0.05,
                value=0.0,
                label=t("training.lora_dropout"),
                info=t("training.lora_dropout_info"),
                elem_classes=["has-info-container"],
            )

            lora_target_mlp = gr.Checkbox(
                label="Target MLP",
                value=False,
                info=(
                    "Also applies LoRA/DoRA to decoder MLP layers "
                    "(gate_proj, up_proj, down_proj). This increases trainable "
                    "capacity and VRAM use; leave off for the legacy attention-only path."
                ),
                elem_classes=["has-info-container"],
            )

    return {
        "training_tensor_dir": training_tensor_dir,
        "training_tensor_dir_browse_btn": training_tensor_dir_browse_btn,
        "load_dataset_btn": load_dataset_btn,
        "training_dataset_info": training_dataset_info,
        "lora_model_config": model_controls["lora_model_config"],
        "lora_vram_preset": lora_vram_preset,
        "lora_adapter_type": lora_adapter_type,
        "lora_adapter_info": lora_adapter_info,
        "lora_name": lora_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_mlp": lora_target_mlp,
    }
