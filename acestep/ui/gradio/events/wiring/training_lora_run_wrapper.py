"""LoRA training stream wrapper for Gradio run controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterator

from loguru import logger

from .. import training_handlers as train_h
from ..training.subprocess_init import build_dit_init_payload
from ..training.subprocess_lora_training import stream_lora_training_subprocess


def build_lora_training_wrapper(
    dit_handler: Any,
    *,
    normalize_training_state: Callable[[Any], dict[str, bool]],
    sample_setting_keys: tuple[str, ...] = (),
) -> Callable[..., Iterator[tuple[Any, Any, Any, dict[str, bool]]]]:
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
        sample_seed: Any,
        sample_output_dir: Any,
        sample_offload_training_model: Any,
        sample_offload_generation: Any,
        training_subprocess: Any,
        model_config: Any,
        vram_preset: Any,
        training_state: Any,
        *sample_setting_values: Any,
    ) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
        """Stream LoRA training progress and normalize failure outputs for UI."""

        state = normalize_training_state(training_state)
        try:
            sample_generation_settings = _sample_generation_settings_from_values(
                sample_setting_keys,
                sample_setting_values,
            )
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
                "sample_duration": sample_generation_settings.get(
                    "audio_duration",
                    -1.0,
                ),
                "sample_inference_steps": sample_generation_settings.get(
                    "inference_steps",
                    8,
                ),
                "sample_seed": sample_seed,
                "sample_output_dir": sample_output_dir,
                "sample_offload_training_model": sample_offload_training_model,
                "sample_offload_generation": sample_offload_generation,
                "model_config": model_config,
                "vram_preset": vram_preset,
                "sample_generation_model_config": sample_generation_settings.get(
                    "config_path"
                ),
                "sample_generation_settings": sample_generation_settings,
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
            yield from _stream_inline_training(dit_handler, state, training_args)
        except Exception as exc:  # pragma: no cover - defensive UI wrapper
            logger.exception("Training wrapper error")
            yield f"\u274c Error: {exc!s}", f"{exc!s}", None, state

    return training_wrapper


def _stream_inline_training(
    dit_handler: Any,
    state: dict[str, bool],
    training_args: dict[str, Any],
) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
    """Stream same-process LoRA training for users who disable subprocess mode."""

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


def _sample_generation_settings_from_values(
    keys: tuple[str, ...],
    values: tuple[Any, ...],
) -> dict[str, Any]:
    """Build a JSON-safe settings dictionary from ordered generation inputs."""

    settings: dict[str, Any] = {}
    for key, value in zip(keys, values):
        settings[key] = value
    return settings
