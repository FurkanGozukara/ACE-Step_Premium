"""LoRA training-tab facade for the Gradio training interface."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.interfaces.training_lora_tab_dataset import (
    build_lora_dataset_and_adapter_controls,
)
from acestep.ui.gradio.interfaces.training_lora_tab_guide import (
    build_lora_training_guide,
)
from acestep.ui.gradio.interfaces.training_lora_tab_run_export import (
    build_lora_run_and_export_controls,
)
from acestep.ui.gradio.interfaces.training_lora_tab_samples import (
    build_lora_sample_generation_controls,
)
from acestep.ui.gradio.interfaces.training_lora_tab_vram import (
    build_lora_vram_controls,
)


def create_training_lora_tab(
    *,
    epoch_min: int,
    epoch_step: int,
    epoch_default: int,
    dit_handler=None,
    init_params: dict | None = None,
) -> dict[str, object]:
    """Create the LoRA training tab and return component handles for wiring."""

    with gr.Tab(t("training.tab_train_lora")):
        create_help_button("training_train")
        build_lora_training_guide()
        tab_controls: dict[str, object] = {}
        tab_controls.update(
            build_lora_dataset_and_adapter_controls(
                dit_handler=dit_handler,
                init_params=init_params,
            )
        )
        tab_controls.update(build_lora_vram_controls())
        selected_model = getattr(tab_controls["lora_model_config"], "value", None)
        tab_controls.update(
            build_lora_run_and_export_controls(
                epoch_min=epoch_min,
                epoch_step=epoch_step,
                epoch_default=epoch_default,
                model_config=selected_model,
            )
        )
        tab_controls.update(build_lora_sample_generation_controls())
    return tab_controls
