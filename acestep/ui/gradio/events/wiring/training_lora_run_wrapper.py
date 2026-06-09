"""LoRA training stream wrapper for Gradio run controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterator

from loguru import logger

from .. import training_handlers as train_h
from ..training.runtime_cleanup import prepare_parent_runtime_for_training
from ..training.subprocess_init import build_dit_init_payload
from ..training.subprocess_lora_training import stream_lora_training_subprocess
from .training_lora_streaming import (
    sample_generation_settings_from_values,
    stream_inline_lora_training,
)


def build_lora_training_wrapper(
    dit_handler: Any,
    *,
    normalize_training_state: Callable[[Any], dict[str, bool]],
    sample_setting_keys: tuple[str, ...] = (),
    llm_handler: Any = None,
) -> Callable[..., Iterator[tuple[Any, Any, Any, dict[str, bool]]]]:
    """Build the training stream wrapper bound to the current DiT handler."""

    def training_wrapper(
        tensor_dir: Any,
        lora_name: Any,
        adapter_type: Any,
        lora_rank: Any,
        lora_alpha: Any,
        lora_dropout: Any,
        target_mlp: Any,
        learning_rate: Any,
        train_epochs: Any,
        train_batch_size: Any,
        gradient_accumulation: Any,
        save_every_n_epochs: Any,
        save_best: Any,
        save_best_after: Any,
        save_best_smoothing_window: Any,
        save_best_min_delta: Any,
        training_shift: Any,
        training_num_inference_steps: Any,
        training_seed: Any,
        optimizer_type: Any,
        scheduler_type: Any,
        timestep_mode: Any,
        adaptive_timestep_ratio: Any,
        validation_split_percent: Any,
        lora_output_dir: Any,
        resume_checkpoint_dir: Any,
        gradient_checkpointing: Any,
        activation_cpu_offload: Any,
        offload_non_decoder: Any,
        keep_frozen_base_in_compute_dtype: Any,
        compile_model: Any,
        base_quantization: Any,
        empty_cache_every_n_steps: Any,
        sample_generation_enabled: Any,
        sample_every_n_epochs: Any,
        sample_prompt: Any,
        sample_lyrics: Any,
        sample_seed: Any,
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
            sample_generation_settings = sample_generation_settings_from_values(
                sample_setting_keys,
                sample_setting_values,
            )
            selected_optimizer = str(optimizer_type or "").strip().casefold()
            training_args = {
                "tensor_dir": tensor_dir,
                "adapter_type": adapter_type,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_mlp": target_mlp,
                "learning_rate": learning_rate,
                "train_epochs": train_epochs,
                "train_batch_size": train_batch_size,
                "gradient_accumulation": gradient_accumulation,
                "save_every_n_epochs": save_every_n_epochs,
                "save_best": save_best,
                "save_best_after": save_best_after,
                "save_best_smoothing_window": save_best_smoothing_window,
                "save_best_min_delta": save_best_min_delta,
                "training_shift": training_shift,
                "training_num_inference_steps": training_num_inference_steps,
                "training_seed": training_seed,
                "optimizer_type": optimizer_type,
                "scheduler_type": scheduler_type,
                "timestep_mode": timestep_mode,
                "adaptive_timestep_ratio": adaptive_timestep_ratio,
                "validation_split_percent": validation_split_percent,
                "lora_output_dir": lora_output_dir,
                "resume_checkpoint_dir": resume_checkpoint_dir,
                "lora_name": str(lora_name or ""),
                "gradient_checkpointing": gradient_checkpointing,
                "activation_cpu_offload": activation_cpu_offload,
                "offload_non_decoder": offload_non_decoder,
                "keep_frozen_base_in_compute_dtype": keep_frozen_base_in_compute_dtype,
                "compile_model": compile_model,
                "use_8bit_adam": selected_optimizer == "adamw8bit",
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
                dit_init_params = build_dit_init_payload(
                    dit_handler,
                    model_config,
                    training_safe=True,
                )
                cleanup_status = prepare_parent_runtime_for_training(
                    dit_handler,
                    llm_handler,
                    release_dit=True,
                )
                if cleanup_status:
                    yield cleanup_status, "", None, state
                yield from stream_lora_training_subprocess(
                    dit_init_params=dit_init_params,
                    training_args=training_args,
                    training_state=state,
                )
                return
            cleanup_status = prepare_parent_runtime_for_training(
                dit_handler,
                llm_handler,
                release_dit=False,
            )
            if cleanup_status:
                yield cleanup_status, "", None, state
            yield from stream_inline_lora_training(
                train_h.start_training,
                dit_handler,
                state,
                training_args,
            )
        except Exception as exc:  # pragma: no cover - defensive UI wrapper
            logger.exception("Training wrapper error")
            yield f"\u274c Error: {exc!s}", f"{exc!s}", None, state

    return training_wrapper
