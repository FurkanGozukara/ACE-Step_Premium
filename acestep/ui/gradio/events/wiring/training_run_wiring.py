"""Training run wiring helpers extracted from ``events.__init__``."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.events.local_path_dialogs import (
    normalize_dialog_path,
    select_folder_path,
)

from .. import training_handlers as train_h
from .context import TrainingWiringContext
from .training_lora_run_wrapper import build_lora_training_wrapper
from .training_lokr_wiring import register_lokr_training_handlers
from .training_sample_generation_settings import (
    sample_generation_input_components,
    sample_generation_setting_keys,
)
from ...interfaces.training_lora_vram_presets import lora_vram_preset_updates


def _normalize_training_state(training_state: Any) -> dict[str, bool]:
    """Return a valid mutable training-state mapping for streaming wrappers."""

    if isinstance(training_state, dict):
        return training_state
    return {"is_training": False, "should_stop": False}


def register_training_run_handlers(context: TrainingWiringContext) -> None:
    """Register training run-tab handlers with stable IO ordering."""

    training_section = context.training_section
    sample_setting_keys = sample_generation_setting_keys(context.generation_section)
    sample_setting_inputs = sample_generation_input_components(context.generation_section)
    training_wrapper = build_lora_training_wrapper(
        context.dit_handler,
        normalize_training_state=_normalize_training_state,
        sample_setting_keys=sample_setting_keys,
    )

    def browse_and_load_training_dataset(current_path: str):
        """Pick a tensor folder and immediately load its dataset summary."""

        selected = select_folder_path(current_path)
        if not selected or selected == normalize_dialog_path(current_path):
            return gr.update(), "No tensor folder selected."
        return gr.update(value=selected), train_h.load_training_dataset(selected)

    # ========== Training Tab Handlers ==========
    training_section["training_tensor_dir_browse_btn"].click(
        fn=browse_and_load_training_dataset,
        inputs=[training_section["training_tensor_dir"]],
        outputs=[
            training_section["training_tensor_dir"],
            training_section["training_dataset_info"],
        ],
    )

    training_section["load_dataset_btn"].click(
        fn=train_h.load_training_dataset,
        inputs=[training_section["training_tensor_dir"]],
        outputs=[training_section["training_dataset_info"]],
    )

    training_section["lora_vram_preset"].change(
        fn=lora_vram_preset_updates,
        inputs=[training_section["lora_vram_preset"]],
        outputs=[
            training_section["lora_rank"],
            training_section["lora_alpha"],
            training_section["lora_gradient_checkpointing"],
            training_section["lora_activation_cpu_offload"],
            training_section["lora_offload_non_decoder"],
            training_section["lora_keep_frozen_bf16"],
            training_section["lora_use_8bit_adam"],
            training_section["lora_base_quantization"],
            training_section["lora_empty_cache_every_n_steps"],
        ],
    )

    training_section["lora_output_dir_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["lora_output_dir"]],
        outputs=[training_section["lora_output_dir"]],
    )

    training_section["resume_checkpoint_dir_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["resume_checkpoint_dir"]],
        outputs=[training_section["resume_checkpoint_dir"]],
    )

    training_section["export_path_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["export_path"]],
        outputs=[training_section["export_path"]],
    )

    training_section["start_training_btn"].click(
        fn=training_wrapper,
        inputs=[
            training_section["training_tensor_dir"],
            training_section["lora_name"],
            training_section["lora_rank"],
            training_section["lora_alpha"],
            training_section["lora_dropout"],
            training_section["learning_rate"],
            training_section["train_epochs"],
            training_section["train_batch_size"],
            training_section["gradient_accumulation"],
            training_section["save_every_n_epochs"],
            training_section["training_shift"],
            training_section["training_seed"],
            training_section["lora_output_dir"],
            training_section["resume_checkpoint_dir"],
            training_section["lora_gradient_checkpointing"],
            training_section["lora_activation_cpu_offload"],
            training_section["lora_offload_non_decoder"],
            training_section["lora_keep_frozen_bf16"],
            training_section["lora_use_8bit_adam"],
            training_section["lora_base_quantization"],
            training_section["lora_empty_cache_every_n_steps"],
            training_section["lora_sample_enabled"],
            training_section["lora_sample_every_n_epochs"],
            training_section["lora_sample_prompt"],
            training_section["lora_sample_lyrics"],
            training_section["lora_sample_duration"],
            training_section["lora_sample_inference_steps"],
            training_section["lora_sample_seed"],
            training_section["lora_sample_output_dir"],
            training_section["lora_sample_offload_training_model"],
            training_section["lora_sample_offload_generation"],
            training_section["training_subprocess"],
            training_section["lora_model_config"],
            training_section["lora_vram_preset"],
            training_section["training_state"],
            *sample_setting_inputs,
        ],
        outputs=[
            training_section["training_progress"],
            training_section["training_log"],
            training_section["training_loss_plot"],
            training_section["training_state"],
        ],
    )

    training_section["stop_training_btn"].click(
        fn=train_h.stop_training,
        inputs=[training_section["training_state"]],
        outputs=[
            training_section["training_progress"],
            training_section["training_state"],
        ],
    )

    training_section["export_lora_btn"].click(
        fn=train_h.export_lora,
        inputs=[
            training_section["export_path"],
            training_section["lora_output_dir"],
        ],
        outputs=[training_section["export_status"]],
    )

    register_lokr_training_handlers(
        context,
        normalize_training_state=_normalize_training_state,
    )
