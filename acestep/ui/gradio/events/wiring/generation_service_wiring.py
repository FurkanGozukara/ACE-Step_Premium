"""Generation service-layer event wiring helpers.

This module contains wiring related to service initialization, LoRA controls,
auto-checkbox controls, and visibility updates for generation components.
"""

from typing import Any

import gradio as gr

from .. import generation_handlers as gen_h
from ..generation.quantization import default_quantization_value
from ...premium_features import (
    SIMPLE_MODEL_ALIASES,
    SIMPLE_MODEL_VALUES,
    model_quality_defaults,
    normalize_simple_model_dropdown_value,
    open_outputs_folder,
)
from ...i18n import get_i18n, reset_language_context, set_language_context
from .context import (
    GenerationWiringContext,
    build_auto_checkbox_inputs,
    build_auto_checkbox_outputs,
)
from .dataset_import_wiring import register_dataset_import_handlers
from .model_default_updates import build_advanced_model_reset_updates


def register_generation_service_handlers(
    context: GenerationWiringContext,
) -> tuple[list[Any], list[Any]]:
    """Register generation service/init handlers and return auto-checkbox lists."""

    generation_section = context.generation_section
    results_section = context.results_section
    dit_handler = context.dit_handler
    llm_handler = context.llm_handler

    # ========== Dataset Handlers ==========
    register_dataset_import_handlers(context)

    # ========== Service Initialization ==========
    generation_section["refresh_btn"].click(
        fn=lambda: gen_h.refresh_checkpoints(dit_handler),
        outputs=[generation_section["checkpoint_dropdown"]],
    )

    generation_section["language_dropdown"].change(
        fn=lambda language: _apply_runtime_language(language),
        inputs=[generation_section["language_dropdown"]],
        outputs=[generation_section["language_dropdown"]],
    )

    def _set_legacy_cfg_prompt(enabled):
        llm_handler.use_legacy_cfg_prompt = bool(enabled)

    generation_section["lm_use_legacy_cfg_prompt"].change(
        fn=_set_legacy_cfg_prompt,
        inputs=[generation_section["lm_use_legacy_cfg_prompt"]],
        outputs=[],
    )

    model_type_outputs = [
        generation_section["inference_steps"],
        generation_section["guidance_scale"],
        generation_section["use_adg"],
        generation_section["shift"],
        generation_section["cfg_interval_start"],
        generation_section["cfg_interval_end"],
        generation_section["task_type"],
        generation_section["generation_mode"],
        generation_section["init_llm_checkbox"],
        generation_section["think_checkbox"],
        generation_section["generate_lm_audio_codes"],
        generation_section["allow_lm_batch"],
        generation_section["use_cot_metas"],
        generation_section["use_cot_caption"],
        generation_section["use_cot_language"],
        generation_section["dcw_enabled"],
        generation_section["dcw_mode"],
        generation_section["dcw_scaler"],
        generation_section["dcw_high_scaler"],
    ]
    advanced_model_reset_outputs = [
        generation_section["infer_method"],
        generation_section["sampler_mode"],
        generation_section["velocity_norm_threshold"],
        generation_section["velocity_ema_factor"],
        generation_section["custom_timesteps"],
        generation_section["dcw_wavelet"],
    ]
    if "simple_model_dropdown" in generation_section:
        generation_section["config_path"].change(
            fn=_apply_config_path_change_with_simple_sync,
            inputs=[
                generation_section["config_path"],
                generation_section["generation_mode"],
            ],
            outputs=[
                *model_type_outputs,
                generation_section["simple_model_dropdown"],
                *advanced_model_reset_outputs,
            ],
            show_progress="hidden",
        )
    else:
        generation_section["config_path"].change(
            fn=_apply_config_path_change_with_advanced_resets,
            inputs=[
                generation_section["config_path"],
                generation_section["generation_mode"],
            ],
            outputs=[
                *model_type_outputs,
                *advanced_model_reset_outputs,
            ],
            show_progress="hidden",
        )

    if "simple_quantization" in generation_section:
        generation_section["quantization_checkbox"].change(
            fn=lambda quantization: gr.update(
                value=default_quantization_value(quantization)
            ),
            inputs=[generation_section["quantization_checkbox"]],
            outputs=[generation_section["simple_quantization"]],
        )

    # ========== Tier Override ==========
    tier_outputs = [
        generation_section["offload_to_cpu_checkbox"],
        generation_section["offload_dit_to_cpu_checkbox"],
        generation_section["compile_model_checkbox"],
        generation_section["quantization_checkbox"],
        generation_section["backend_dropdown"],
        generation_section["lm_model_path"],
        generation_section["init_llm_checkbox"],
        generation_section["batch_size_input"],
        generation_section["audio_duration"],
        generation_section["gpu_info_display"],
    ]
    if "simple_quantization" in generation_section:
        generation_section["tier_dropdown"].change(
            fn=lambda tier: _apply_tier_change_with_simple_quantization(
                tier,
                llm_handler,
            ),
            inputs=[generation_section["tier_dropdown"]],
            outputs=[
                tier_outputs[0],
                tier_outputs[1],
                tier_outputs[2],
                tier_outputs[3],
                generation_section["simple_quantization"],
                tier_outputs[4],
                tier_outputs[5],
                tier_outputs[6],
                tier_outputs[7],
                tier_outputs[8],
                tier_outputs[9],
            ],
        )
    else:
        generation_section["tier_dropdown"].change(
            fn=lambda tier: gen_h.on_tier_change(tier, llm_handler),
            inputs=[generation_section["tier_dropdown"]],
            outputs=tier_outputs,
        )

    init_event = generation_section["init_btn"].click(
        fn=lambda *args: gen_h.init_service_wrapper(dit_handler, llm_handler, *args),
        inputs=[
            generation_section["checkpoint_dropdown"],
            generation_section["config_path"],
            generation_section["device"],
            generation_section["init_llm_checkbox"],
            generation_section["lm_model_path"],
            generation_section["backend_dropdown"],
            generation_section["use_flash_attention_checkbox"],
            generation_section["offload_to_cpu_checkbox"],
            generation_section["offload_dit_to_cpu_checkbox"],
            generation_section["compile_model_checkbox"],
            generation_section["quantization_checkbox"],
            generation_section["mlx_dit_checkbox"],
            generation_section["generation_mode"],
            generation_section["batch_size_input"],
            generation_section["think_checkbox"],
            generation_section["vae_checkpoint"],
        ],
        outputs=[
            generation_section["init_status"],
            generation_section["generate_btn"],
            generation_section["service_config_accordion"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["use_adg"],
            generation_section["shift"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["task_type"],
            generation_section["generation_mode"],
            generation_section["init_llm_checkbox"],
            generation_section["dcw_enabled"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["think_checkbox"],
        ],
        show_progress_on=[generation_section["init_status"]],
    )
    init_event.then(
        fn=_apply_dcw_defaults_for_model,
        inputs=[generation_section["config_path"]],
        outputs=[
            generation_section["dcw_enabled"],
            generation_section["dcw_mode"],
            generation_section["dcw_scaler"],
            generation_section["dcw_high_scaler"],
        ],
    )

    # ========== LoRA Handlers ==========
    generation_section["refresh_lora_dropdown_btn"].click(
        fn=gen_h.refresh_lora_dropdown,
        inputs=[
            generation_section["lora_dropdown"],
            generation_section["lora_path"],
        ],
        outputs=[
            generation_section["lora_dropdown"],
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
        ],
    )

    generation_section["lora_dropdown"].change(
        fn=gen_h.select_lora_dropdown_path,
        inputs=[generation_section["lora_dropdown"]],
        outputs=[
            generation_section["lora_path"],
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
        ],
    )

    generation_section["lora_path"].change(
        fn=gen_h.update_lora_next_run_status,
        inputs=[
            generation_section["lora_path"],
            generation_section["lora_dropdown"],
        ],
        outputs=[
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
        ],
    )

    generation_section["lora_scale_slider"].change(
        fn=gen_h.update_lora_next_run_status,
        inputs=[
            generation_section["lora_path"],
            generation_section["lora_dropdown"],
        ],
        outputs=[
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
        ],
    )

    # ========== MLX VAE Chunk Size ==========
    generation_section["mlx_vae_chunk_size"].change(
        fn=lambda val: setattr(dit_handler, "mlx_vae_chunk_size", int(val)),
        inputs=[generation_section["mlx_vae_chunk_size"]],
    )

    # ========== Auto Checkbox Handlers ==========
    auto_field_map = {
        "bpm_auto": ("bpm", "bpm"),
        "key_auto": ("key_scale", "key_scale"),
        "timesig_auto": ("time_signature", "time_signature"),
        "vocal_lang_auto": ("vocal_language", "vocal_language"),
        "duration_auto": ("audio_duration", "audio_duration"),
    }
    for auto_key, (field_name, comp_key) in auto_field_map.items():
        generation_section[auto_key].change(
            fn=lambda checked, fn=field_name: gen_h.on_auto_checkbox_change(checked, fn),
            inputs=[generation_section[auto_key]],
            outputs=[generation_section[comp_key]],
        )

    auto_checkbox_outputs = build_auto_checkbox_outputs(context)
    auto_checkbox_inputs = build_auto_checkbox_inputs(context)

    generation_section["reset_all_auto_btn"].click(
        fn=gen_h.reset_all_auto,
        outputs=auto_checkbox_outputs,
    )
    generation_section["open_outputs_folder_btn"].click(
        fn=open_outputs_folder,
        outputs=[results_section["status_output"]],
    )

    # ========== UI Visibility Updates ==========
    generation_section["init_llm_checkbox"].change(
        fn=gen_h.update_negative_prompt_visibility,
        inputs=[generation_section["init_llm_checkbox"]],
        outputs=[generation_section["lm_negative_prompt"]],
    )

    generation_section["batch_size_input"].change(
        fn=gen_h.update_audio_components_visibility,
        inputs=[generation_section["batch_size_input"]],
        outputs=[
            results_section["audio_col_1"],
            results_section["audio_col_2"],
            results_section["audio_col_3"],
            results_section["audio_col_4"],
            results_section["audio_row_5_8"],
            results_section["audio_col_5"],
            results_section["audio_col_6"],
            results_section["audio_col_7"],
            results_section["audio_col_8"],
        ],
    )

    return auto_checkbox_inputs, auto_checkbox_outputs


def _apply_runtime_language(language: str) -> dict[str, Any]:
    """Update i18n language at the Gradio request boundary.

    Sets a per-request ``ContextVar`` so any ``t()`` calls within this
    handler use *language*, then updates the shared instance default so
    future requests without an explicit context inherit it.  The
    ``ContextVar`` is reset on exit to avoid poisoning reused
    thread-pool workers with a stale language value.

    Args:
        language: Selected UI language code from the language dropdown.

    Returns:
        A ``gr.update`` payload preserving the selected dropdown value.
    """
    # Set ContextVar for this handler's scope.  No t() calls happen here
    # today, but the pattern establishes the request-boundary convention
    # for future handlers that adopt per-request language isolation.
    token = set_language_context(language)
    try:
        get_i18n(language)
        return gr.update(value=language)
    finally:
        reset_language_context(token)


def _apply_config_path_change_with_simple_sync(
    config_path: str | None,
    current_mode: str | None = None,
) -> tuple[Any, ...]:
    """Update model controls, mirror simple selector, and reset advanced defaults."""

    model_and_behavior_updates = _apply_config_path_change(config_path, current_mode)
    selected = str(config_path or "").strip()
    simple_model_update = (
        gr.update(value=normalize_simple_model_dropdown_value(selected))
        if selected in SIMPLE_MODEL_VALUES or selected in SIMPLE_MODEL_ALIASES
        else gr.update()
    )
    return (
        *model_and_behavior_updates,
        simple_model_update,
        *build_advanced_model_reset_updates(config_path),
    )


def _apply_config_path_change_with_advanced_resets(
    config_path: str | None,
    current_mode: str | None = None,
) -> tuple[Any, ...]:
    """Update model controls and reset advanced-only model defaults."""

    return (
        *_apply_config_path_change(config_path, current_mode),
        *build_advanced_model_reset_updates(config_path),
    )


def _apply_config_path_change(
    config_path: str | None,
    current_mode: str | None = None,
) -> tuple[Any, ...]:
    """Update model-type controls and all LM/DCW behavior controls."""

    model_updates = list(gen_h.update_model_type_settings(config_path, current_mode))
    quality_defaults = model_quality_defaults(config_path)
    model_updates[8] = gr.update(value=quality_defaults["init_lm_checkbox"])
    behavior_updates = (
        gr.update(value=quality_defaults["think_checkbox"]),
        gr.update(value=quality_defaults["generate_lm_audio_codes"]),
        gr.update(value=quality_defaults["allow_lm_batch"]),
        gr.update(value=quality_defaults["use_cot_metas"]),
        gr.update(value=quality_defaults["use_cot_caption"]),
        gr.update(value=quality_defaults["use_cot_language"]),
        gr.update(value=quality_defaults["dcw_enabled"]),
        gr.update(value=quality_defaults["dcw_mode"]),
        gr.update(value=quality_defaults["dcw_scaler"]),
        gr.update(value=quality_defaults["dcw_high_scaler"]),
    )
    return (*model_updates, *behavior_updates)


def _apply_dcw_defaults_for_model(config_path: str | None) -> tuple[Any, ...]:
    """Return DCW enable/mode/scaler defaults for the selected model."""

    quality_defaults = model_quality_defaults(config_path)
    return (
        gr.update(value=quality_defaults["dcw_enabled"]),
        gr.update(value=quality_defaults["dcw_mode"]),
        gr.update(value=quality_defaults["dcw_scaler"]),
        gr.update(value=quality_defaults["dcw_high_scaler"]),
    )


def _apply_tier_change_with_simple_quantization(
    tier: str | None,
    llm_handler: Any,
) -> tuple[Any, ...]:
    """Apply a GPU tier and mirror its quantization default into the simple tab."""

    updates = gen_h.on_tier_change(tier, llm_handler)
    if len(updates) != 10:
        return tuple(gr.update() for _ in range(11))
    return (
        updates[0],
        updates[1],
        updates[2],
        updates[3],
        updates[3],
        updates[4],
        updates[5],
        updates[6],
        updates[7],
        updates[8],
        updates[9],
    )
