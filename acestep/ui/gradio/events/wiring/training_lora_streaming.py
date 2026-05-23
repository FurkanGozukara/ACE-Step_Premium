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
