"""Training run wiring helpers extracted from ``events.__init__``."""

from typing import Any, Iterator

import gradio as gr
from loguru import logger

from acestep.ui.gradio.events.local_path_dialogs import (
    normalize_dialog_path,
    select_folder_path,
)

from .. import training_handlers as train_h
from ..training.subprocess_init import build_dit_init_payload
from ..training.subprocess_lora_training import stream_lora_training_subprocess
from .context import TrainingWiringContext
from .training_lokr_wiring import register_lokr_training_handlers


def _normalize_training_state(training_state: Any) -> dict[str, bool]:
    """Return a valid mutable training-state mapping for streaming wrappers."""

    if isinstance(training_state, dict):
        return training_state
    return {"is_training": False, "should_stop": False}


def _build_training_wrapper(dit_handler: Any):
    """Build the training stream wrapper bound to the current DiT handler."""

    def training_wrapper(
        tensor_dir: Any,
        lora_name: Any,
        lora_rank: Any,
        lora_alpha: Any,
        lora_dropout: Any,
        learning_rate: Any,
        train_epochs: Any,
        train_batch_size: Any,
        gradient_accumulation: Any,
        save_every_n_epochs: Any,
        training_shift: Any,
        training_seed: Any,
        lora_output_dir: Any,
        resume_checkpoint_dir: Any,
        gradient_checkpointing: Any,
        activation_cpu_offload: Any,
        offload_non_decoder: Any,
        keep_frozen_base_in_compute_dtype: Any,
        use_8bit_adam: Any,
        base_quantization: Any,
        empty_cache_every_n_steps: Any,
        sample_generation_enabled: Any,
        sample_every_n_epochs: Any,
        sample_prompt: Any,
        sample_lyrics: Any,
        sample_duration: Any,
        sample_inference_steps: Any,
        sample_seed: Any,
        sample_output_dir: Any,
        sample_offload_training_model: Any,
        sample_offload_generation: Any,
        training_subprocess: Any,
        model_config: Any,
        training_state: Any,
    ) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
        """Stream LoRA training progress and normalize failure outputs for UI."""

        state = _normalize_training_state(training_state)
        try:
            training_args = {
                "tensor_dir": tensor_dir,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "train_epochs": train_epochs,
                "train_batch_size": train_batch_size,
                "gradient_accumulation": gradient_accumulation,
                "save_every_n_epochs": save_every_n_epochs,
                "training_shift": training_shift,
                "training_seed": training_seed,
                "lora_output_dir": lora_output_dir,
                "resume_checkpoint_dir": resume_checkpoint_dir,
                "lora_name": str(lora_name or ""),
                "gradient_checkpointing": gradient_checkpointing,
                "activation_cpu_offload": activation_cpu_offload,
                "offload_non_decoder": offload_non_decoder,
                "keep_frozen_base_in_compute_dtype": keep_frozen_base_in_compute_dtype,
                "use_8bit_adam": use_8bit_adam,
                "base_quantization": base_quantization,
                "empty_cache_every_n_steps": empty_cache_every_n_steps,
                "sample_generation_enabled": sample_generation_enabled,
                "sample_every_n_epochs": sample_every_n_epochs,
                "sample_prompt": sample_prompt,
                "sample_lyrics": sample_lyrics,
                "sample_duration": sample_duration,
                "sample_inference_steps": sample_inference_steps,
                "sample_seed": sample_seed,
                "sample_output_dir": sample_output_dir,
                "sample_offload_training_model": sample_offload_training_model,
                "sample_offload_generation": sample_offload_generation,
                "model_config": model_config,
            }
            if training_subprocess:
                yield from stream_lora_training_subprocess(
                    dit_init_params=build_dit_init_payload(
                        dit_handler,
                        model_config,
                        training_safe=True,
                    ),
                    training_args=training_args,
                    training_state=state,
                )
                return
            for progress, log_msg, plot, next_state in train_h.start_training(
                training_args["tensor_dir"],
                dit_handler,
                training_args["lora_rank"],
                training_args["lora_alpha"],
                training_args["lora_dropout"],
                training_args["learning_rate"],
                training_args["train_epochs"],
                training_args["train_batch_size"],
                training_args["gradient_accumulation"],
                training_args["save_every_n_epochs"],
                training_args["training_shift"],
                training_args["training_seed"],
                training_args["lora_output_dir"],
                training_args["resume_checkpoint_dir"],
                state,
                lora_name=training_args["lora_name"],
                gradient_checkpointing=training_args["gradient_checkpointing"],
                activation_cpu_offload=training_args["activation_cpu_offload"],
                offload_non_decoder=training_args["offload_non_decoder"],
                keep_frozen_base_in_compute_dtype=training_args[
                    "keep_frozen_base_in_compute_dtype"
                ],
                use_8bit_adam=training_args["use_8bit_adam"],
                base_quantization=training_args["base_quantization"],
                empty_cache_every_n_steps=training_args["empty_cache_every_n_steps"],
                sample_generation_enabled=training_args["sample_generation_enabled"],
                sample_every_n_epochs=training_args["sample_every_n_epochs"],
                sample_prompt=training_args["sample_prompt"],
                sample_lyrics=training_args["sample_lyrics"],
                sample_duration=training_args["sample_duration"],
                sample_inference_steps=training_args["sample_inference_steps"],
                sample_seed=training_args["sample_seed"],
                sample_output_dir=training_args["sample_output_dir"],
                sample_offload_training_model=training_args[
                    "sample_offload_training_model"
                ],
                sample_offload_generation=training_args["sample_offload_generation"],
                model_config=training_args["model_config"],
            ):
                yield progress, log_msg, plot, next_state
        except Exception as exc:  # pragma: no cover - defensive UI wrapper
            logger.exception("Training wrapper error")
            yield f"\u274c Error: {exc!s}", f"{exc!s}", None, state

    return training_wrapper


def register_training_run_handlers(context: TrainingWiringContext) -> None:
    """Register training run-tab handlers with stable IO ordering."""

    training_section = context.training_section
    training_wrapper = _build_training_wrapper(context.dit_handler)

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
            training_section["training_state"],
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
