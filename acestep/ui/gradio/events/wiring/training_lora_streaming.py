"""LoRA training stream helpers shared by the Gradio run wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterator


def stream_inline_lora_training(
    start_training_fn: Callable[..., Iterator[tuple[Any, Any, Any, dict[str, bool]]]],
    dit_handler: Any,
    state: dict[str, bool],
    training_args: dict[str, Any],
) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
    """Stream same-process LoRA training for users who disable subprocess mode."""

    for progress, log_msg, plot, next_state in start_training_fn(
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
        training_num_inference_steps=training_args["training_num_inference_steps"],
        lora_name=training_args["lora_name"],
        adapter_type=training_args["adapter_type"],
        target_mlp=training_args["target_mlp"],
        gradient_checkpointing=training_args["gradient_checkpointing"],
        activation_cpu_offload=training_args["activation_cpu_offload"],
        offload_non_decoder=training_args["offload_non_decoder"],
        keep_frozen_base_in_compute_dtype=training_args[
            "keep_frozen_base_in_compute_dtype"
        ],
        compile_model=training_args["compile_model"],
        use_8bit_adam=training_args["use_8bit_adam"],
        optimizer_type=training_args["optimizer_type"],
        weight_decay=training_args["weight_decay"],
        adam_beta1=training_args["adam_beta1"],
        adam_beta2=training_args["adam_beta2"],
        adam_epsilon=training_args["adam_epsilon"],
        adamw8bit_min_8bit_size=training_args["adamw8bit_min_8bit_size"],
        adamw8bit_percentile_clipping=training_args[
            "adamw8bit_percentile_clipping"
        ],
        adamw8bit_block_wise=training_args["adamw8bit_block_wise"],
        adamw8bit_paged=training_args["adamw8bit_paged"],
        adafactor_epsilon1=training_args["adafactor_epsilon1"],
        adafactor_epsilon2=training_args["adafactor_epsilon2"],
        adafactor_clip_threshold=training_args["adafactor_clip_threshold"],
        adafactor_decay_rate=training_args["adafactor_decay_rate"],
        adafactor_beta1=training_args["adafactor_beta1"],
        adafactor_scale_parameter=training_args["adafactor_scale_parameter"],
        adafactor_relative_step=training_args["adafactor_relative_step"],
        adafactor_warmup_init=training_args["adafactor_warmup_init"],
        scheduler_type=training_args["scheduler_type"],
        save_best=training_args["save_best"],
        save_best_after=training_args["save_best_after"],
        save_best_smoothing_window=training_args["save_best_smoothing_window"],
        save_best_min_delta=training_args["save_best_min_delta"],
        timestep_mode=training_args["timestep_mode"],
        adaptive_timestep_ratio=training_args["adaptive_timestep_ratio"],
        validation_split_percent=training_args["validation_split_percent"],
        base_quantization=training_args["base_quantization"],
        empty_cache_every_n_steps=training_args["empty_cache_every_n_steps"],
        sample_generation_enabled=training_args["sample_generation_enabled"],
        sample_every_n_epochs=training_args["sample_every_n_epochs"],
        sample_prompt=training_args["sample_prompt"],
        sample_lyrics=training_args["sample_lyrics"],
        sample_duration=training_args["sample_duration"],
        sample_inference_steps=training_args["sample_inference_steps"],
        sample_seed=training_args["sample_seed"],
        sample_offload_training_model=training_args["sample_offload_training_model"],
        sample_offload_generation=training_args["sample_offload_generation"],
        model_config=training_args["model_config"],
        vram_preset=training_args["vram_preset"],
        sample_generation_model_config=training_args[
            "sample_generation_model_config"
        ],
        sample_generation_settings=training_args["sample_generation_settings"],
    ):
        yield progress, log_msg, plot, next_state


def sample_generation_settings_from_values(
    keys: tuple[str, ...],
    values: tuple[Any, ...],
) -> dict[str, Any]:
    """Build a JSON-safe settings dictionary from ordered generation inputs."""

    settings: dict[str, Any] = {}
    for key, value in zip(keys, values):
        settings[key] = value
    return settings
