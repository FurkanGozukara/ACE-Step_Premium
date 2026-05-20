"""Training run wiring helpers extracted from ``events.__init__``."""

from typing import Any

from acestep.ui.gradio.events.local_path_dialogs import (
    select_folder_path,
    select_pt_file_path,
)

from .. import training_handlers as train_h
from ..training.schedule_defaults import training_schedule_updates_for_model
from .context import TrainingWiringContext
from .training_lora_run_wrapper import build_lora_training_wrapper
from .training_lokr_wiring import register_lokr_training_handlers
from .training_sample_generation_settings import (
    sample_generation_input_components,
    sample_generation_setting_keys,
)
from .training_step_estimate_wiring import (
    attach_lora_step_estimate_update,
    register_lora_step_estimate_handlers,
)
from .training_tensor_browse import browse_and_load_training_dataset
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

    # ========== Training Tab Handlers ==========
    browse_event = training_section["training_tensor_dir_browse_btn"].click(
        fn=browse_and_load_training_dataset,
        inputs=[training_section["training_tensor_dir"]],
        outputs=[
            training_section["training_tensor_dir"],
            training_section["training_dataset_info"],
        ],
    )
    attach_lora_step_estimate_update(browse_event, training_section)

    load_event = training_section["load_dataset_btn"].click(
        fn=train_h.load_training_dataset,
        inputs=[training_section["training_tensor_dir"]],
        outputs=[training_section["training_dataset_info"]],
    )
    attach_lora_step_estimate_update(load_event, training_section)
    register_lora_step_estimate_handlers(training_section)

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

    training_section["lora_model_config"].change(
        fn=training_schedule_updates_for_model,
        inputs=[training_section["lora_model_config"]],
        outputs=[
            training_section["training_shift"],
            training_section["training_num_inference_steps"],
        ],
    )

    training_section["lora_output_dir_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["lora_output_dir"]],
        outputs=[training_section["lora_output_dir"]],
    )

    training_section["lora_open_output_dir_btn"].click(
        fn=train_h.open_lora_output_folder,
        inputs=[
            training_section["lora_output_dir"],
            training_section["lora_name"],
        ],
        outputs=[training_section["training_progress"]],
    )

    training_section["resume_checkpoint_dir_browse_btn"].click(
        fn=select_pt_file_path,
        inputs=[training_section["resume_checkpoint_dir"]],
        outputs=[training_section["resume_checkpoint_dir"]],
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
            training_section["training_num_inference_steps"],
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
            training_section["lora_sample_seed"],
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

    register_lokr_training_handlers(
        context,
        normalize_training_state=_normalize_training_state,
    )
