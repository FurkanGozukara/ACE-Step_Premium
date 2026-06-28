"""Event wiring for the simple first-tab creation flow."""

from __future__ import annotations

from collections.abc import Iterator
import inspect
from typing import Any

import gradio as gr

from acestep.gpu_config import get_global_gpu_config
from .. import generation_handlers as gen_h
from .. import results_handlers as res_h
from ..generation.cancel_actions import (
    CANCEL_CONFIRM_JS,
    request_generation_cancel_from_ui,
)
from ..generation.quantization import default_quantization_value
from ...premium_features import (
    SIMPLE_MODEL_CHOICES,
    model_quality_defaults,
    normalize_simple_model_dropdown_value,
    open_outputs_folder,
)
from .generation_run_wiring import (
    build_clear_audio_outputs,
    build_generation_run_inputs,
    build_generation_run_outputs,
)
from ..results.result_output_contract import STATUS_INDEX
from .inline_result_preview import (
    build_inline_result_outputs,
    clear_inline_result_preview,
    sync_inline_result_preview,
)
from .model_default_updates import build_advanced_model_reset_updates
from .simple_media_outputs import (
    build_simple_generated_files_update,
    build_simple_media_preview,
    clear_simple_generated_files,
    clear_simple_media_preview,
)
from .simple_create_params import prepare_simple_generation
from .simple_lora_wiring import register_simple_lora_sync_handlers


_STATUS_OUTPUT_INDEX = STATUS_INDEX
_SIMPLE_MODEL_LABELS = {value: label for label, value in SIMPLE_MODEL_CHOICES}


def register_simple_create_handlers(
    *,
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
    dit_handler: Any,
    llm_handler: Any,
) -> None:
    """Wire the simple Create tab to the existing generation backend."""

    def generation_wrapper(*args: Any) -> Iterator[tuple[Any, ...]]:
        """Stream generation outputs and route Gradio progress to the simple tab."""

        yield from _stream_simple_generation(
            dit_handler,
            llm_handler,
            args,
        )

    generation_wrapper.__signature__ = build_simple_generation_wrapper_signature()

    simple_page["simple_open_outputs_btn"].click(
        fn=open_outputs_folder,
        inputs=[],
        outputs=[simple_page["simple_status"]],
    )
    register_simple_lora_sync_handlers(
        simple_page=simple_page,
        generation_section=generation_section,
    )
    _register_vocal_language_sync_handlers(
        simple_page=simple_page,
        generation_section=generation_section,
    )
    _register_negative_prompt_sync_handlers(
        simple_page=simple_page,
        generation_section=generation_section,
    )
    _register_sampler_mode_sync_handlers(
        simple_page=simple_page,
        generation_section=generation_section,
    )

    _register_simple_enhance_handlers(
        simple_page=simple_page,
        generation_section=generation_section,
        results_section=results_section,
        llm_handler=llm_handler,
    )

    simple_page["simple_tier_dropdown"].change(
        fn=lambda tier: _apply_simple_tier_change(tier, llm_handler),
        inputs=[simple_page["simple_tier_dropdown"]],
        outputs=[
            generation_section["tier_dropdown"],
            generation_section["offload_to_cpu_checkbox"],
            generation_section["offload_dit_to_cpu_checkbox"],
            generation_section["compile_model_checkbox"],
            generation_section["quantization_checkbox"],
            simple_page["simple_quantization"],
            generation_section["backend_dropdown"],
            generation_section["lm_model_path"],
            generation_section["init_llm_checkbox"],
            generation_section["batch_size_input"],
            generation_section["audio_duration"],
            generation_section["gpu_info_display"],
            simple_page["simple_batch_size"],
            simple_page["simple_duration"],
            generation_section["device"],
            simple_page["simple_status"],
        ],
    )

    # Gradio .change() fires for programmatic dropdown updates too. The model
    # selector should apply defaults only for direct user selections.
    simple_page["simple_model_dropdown"].input(
        fn=_apply_simple_model_change,
        inputs=[
            simple_page["simple_model_dropdown"],
            generation_section["generation_mode"],
        ],
        outputs=[
            generation_section["config_path"],
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
            generation_section["infer_method"],
            generation_section["sampler_mode"],
            generation_section["velocity_norm_threshold"],
            generation_section["velocity_ema_factor"],
            generation_section["custom_timesteps"],
            generation_section["dcw_wavelet"],
            simple_page["simple_status"],
        ],
        show_progress="hidden",
        show_progress_on=[],
    )

    simple_page["simple_quantization"].change(
        fn=lambda quantization: gr.update(
            value=default_quantization_value(quantization)
        ),
        inputs=[simple_page["simple_quantization"]],
        outputs=[generation_section["quantization_checkbox"]],
    )

    simple_page["simple_random_btn"].click(
        fn=gen_h.load_random_simple_description,
        inputs=[],
        outputs=[
            simple_page["simple_caption"],
            simple_page["simple_instrumental"],
            simple_page["simple_vocal_language"],
        ],
    )

    simple_page["simple_generate_btn"].click(
        fn=prepare_simple_generation,
        inputs=build_simple_prepare_inputs(simple_page),
        outputs=build_simple_prepare_outputs(
            generation_section,
            results_section,
            simple_page["simple_status"],
        ),
    ).then(
        fn=res_h.clear_audio_outputs_for_new_generation,
        outputs=build_clear_audio_outputs(results_section),
    ).then(
        fn=clear_inline_result_preview,
        outputs=build_inline_result_outputs(generation_section),
    ).then(
        fn=clear_simple_media_preview,
        outputs=[
            simple_page["simple_latest_audio"],
            simple_page["simple_latest_video"],
        ],
    ).then(
        fn=clear_simple_generated_files,
        outputs=[simple_page["simple_generated_files"]],
    ).then(
        fn=generation_wrapper,
        inputs=build_generation_run_inputs(generation_section, results_section),
        outputs=[
            *build_generation_run_outputs(generation_section, results_section),
            simple_page["simple_status"],
        ],
        show_progress_on=build_simple_generation_progress_targets(simple_page),
    ).then(
        fn=sync_inline_result_preview,
        inputs=[
            results_section["generated_audio_1"],
            results_section["generated_audio_batch"],
            results_section["status_output"],
        ],
        outputs=build_inline_result_outputs(generation_section),
    ).then(
        fn=build_simple_generated_files_update,
        inputs=[results_section["generated_audio_batch"]],
        outputs=[simple_page["simple_generated_files"]],
    ).then(
        fn=build_simple_media_preview,
        inputs=[
            results_section["generated_audio_1"],
            results_section["status_output"],
            simple_page["simple_cover_image"],
            simple_page["simple_video_resolution"],
            results_section["generated_audio_batch"],
        ],
        outputs=[
            simple_page["simple_latest_audio"],
            simple_page["simple_latest_video"],
            simple_page["simple_status"],
            simple_page["simple_generated_files"],
        ],
    )
    simple_cancel_event = simple_page["simple_cancel_generation_btn"].click(
        fn=None,
        inputs=None,
        outputs=[simple_page["simple_cancel_confirmed_state"]],
        js=CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    simple_cancel_event.then(
        fn=request_generation_cancel_from_ui,
        inputs=[
            simple_page["simple_cancel_confirmed_state"],
            generation_section["subprocess_mode_checkbox"],
        ],
        outputs=[simple_page["simple_status"]],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )


def build_simple_prepare_inputs(simple_page: dict[str, Any]) -> list[Any]:
    """Return simple-tab inputs used to prepare a generation request."""

    return [
        simple_page["simple_caption"],
        simple_page["simple_lyrics"],
        simple_page["simple_vocal_language"],
        simple_page["simple_instrumental"],
        simple_page["simple_vocal_gender"],
        simple_page["simple_duration"],
        simple_page["simple_batch_size"],
        simple_page["simple_random_seed"],
        simple_page["simple_seed"],
        simple_page["simple_quantization"],
        simple_page["simple_model_dropdown"],
        simple_page["simple_bpm_state"],
        simple_page["simple_key_scale_state"],
        simple_page["simple_time_signature_state"],
        simple_page["simple_is_format_caption_state"],
        simple_page["simple_negative_prompt"],
        simple_page["simple_sampler_mode"],
    ]


def build_simple_generation_progress_targets(simple_page: dict[str, Any]) -> list[Any]:
    """Return simple-tab components that should display generation progress."""

    return [
        simple_page["simple_latest_audio"],
        simple_page["simple_status"],
    ]


def build_simple_generation_wrapper_signature() -> inspect.Signature:
    """Return the simple generation wrapper signature with handlers pre-bound."""

    parameters = list(
        inspect.signature(res_h.generate_with_batch_management).parameters.values()
    )
    return inspect.Signature(parameters=parameters[2:])


def _stream_simple_generation(
    dit_handler: Any,
    llm_handler: Any,
    args: tuple[Any, ...],
) -> Iterator[tuple[Any, ...]]:
    """Stream backend generation outputs with compact simple-tab status."""

    args = _apply_simple_low_vram_overrides(args)
    for outputs in res_h.generate_with_batch_management(
        dit_handler,
        llm_handler,
        *args,
    ):
        status = _extract_generation_status(outputs)
        yield (*outputs, status)


def _apply_simple_low_vram_overrides(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Apply tier-specific simple-tab overrides at the generation boundary."""

    gpu_config = get_global_gpu_config()
    values = list(args)
    if getattr(gpu_config, "init_lm_default", True) is False:
        for index in (42, 47, 48, 49, 52, 53, 87):
            if index < len(values):
                values[index] = False
    audio_codes_default = getattr(gpu_config, "generate_lm_audio_codes_default", None)
    if audio_codes_default is not None and 100 < len(values):
        values[100] = bool(audio_codes_default)
    dcw_default = getattr(gpu_config, "dcw_enabled_default", None)
    if dcw_default is not None and 32 < len(values):
        values[32] = bool(dcw_default)
        if not values[32]:
            if 34 < len(values):
                values[34] = 0.0
            if 35 < len(values):
                values[35] = 0.0
    return tuple(values)


def build_simple_prepare_outputs(
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
    status_output: Any,
) -> list[Any]:
    """Return outputs updated before simple generation starts."""

    return [
        generation_section["generation_mode"],
        generation_section["task_type"],
        generation_section["captions"],
        generation_section["lyrics"],
        generation_section["vocal_language"],
        generation_section["audio_duration"],
        generation_section["batch_size_input"],
        generation_section["quantization_checkbox"],
        generation_section["init_llm_checkbox"],
        generation_section["think_checkbox"],
        generation_section["allow_lm_batch"],
        generation_section["random_seed_checkbox"],
        generation_section["seed"],
        generation_section["text2music_audio_code_string"],
        generation_section["bpm_auto"],
        generation_section["key_auto"],
        generation_section["timesig_auto"],
        generation_section["vocal_lang_auto"],
        generation_section["duration_auto"],
        generation_section["use_cot_metas"],
        generation_section["use_cot_caption"],
        generation_section["use_cot_language"],
        generation_section["bpm"],
        generation_section["key_scale"],
        generation_section["time_signature"],
        results_section["is_format_caption_state"],
        status_output,
        generation_section["inference_steps"],
        generation_section["guidance_scale"],
        generation_section["use_adg"],
        generation_section["shift"],
        generation_section["cfg_interval_start"],
        generation_section["cfg_interval_end"],
        generation_section["config_path"],
        generation_section["dcw_enabled"],
        generation_section["dcw_mode"],
        generation_section["dcw_scaler"],
        generation_section["dcw_high_scaler"],
        generation_section["infer_method"],
        generation_section["sampler_mode"],
        generation_section["velocity_norm_threshold"],
        generation_section["velocity_ema_factor"],
        generation_section["custom_timesteps"],
        generation_section["dcw_wavelet"],
        generation_section["generate_lm_audio_codes"],
        generation_section["lm_negative_prompt"],
    ]


def _register_vocal_language_sync_handlers(
    *,
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
) -> None:
    """Keep Generate Song and advanced vocal-language dropdowns synchronized."""

    simple_page["simple_vocal_language"].input(
        fn=_sync_vocal_language_value,
        inputs=[simple_page["simple_vocal_language"]],
        outputs=[generation_section["vocal_language"]],
        show_progress="hidden",
        show_progress_on=[],
    )
    generation_section["vocal_language"].input(
        fn=_sync_vocal_language_value,
        inputs=[generation_section["vocal_language"]],
        outputs=[simple_page["simple_vocal_language"]],
        show_progress="hidden",
        show_progress_on=[],
    )


def _sync_vocal_language_value(value: Any) -> Any:
    """Return a dropdown-safe vocal-language value."""

    text = str(value or "").strip()
    return text or "unknown"


def _register_negative_prompt_sync_handlers(
    *,
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
) -> None:
    """Keep Generate Song and advanced negative prompts synchronized."""

    for event_name in ("input", "change"):
        getattr(simple_page["simple_negative_prompt"], event_name)(
            fn=_sync_negative_prompt_value,
            inputs=[simple_page["simple_negative_prompt"]],
            outputs=[generation_section["lm_negative_prompt"]],
            show_progress="hidden",
            show_progress_on=[],
        )
        getattr(generation_section["lm_negative_prompt"], event_name)(
            fn=_sync_negative_prompt_value,
            inputs=[generation_section["lm_negative_prompt"]],
            outputs=[simple_page["simple_negative_prompt"]],
            show_progress="hidden",
            show_progress_on=[],
        )


def _sync_negative_prompt_value(value: Any) -> str:
    """Normalize negative prompt sync values across simple and advanced fields."""

    text = str(value or "").strip()
    return "" if text.upper() == "NO USER INPUT" else text


def _register_sampler_mode_sync_handlers(
    *,
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
) -> None:
    """Keep Generate Song and advanced sampler dropdowns synchronized."""

    for event_name in ("input", "change"):
        getattr(simple_page["simple_sampler_mode"], event_name)(
            fn=_sync_sampler_mode_value,
            inputs=[simple_page["simple_sampler_mode"]],
            outputs=[generation_section["sampler_mode"]],
            show_progress="hidden",
            show_progress_on=[],
        )
        getattr(generation_section["sampler_mode"], event_name)(
            fn=_sync_sampler_mode_value,
            inputs=[generation_section["sampler_mode"]],
            outputs=[simple_page["simple_sampler_mode"]],
            show_progress="hidden",
            show_progress_on=[],
        )


def _sync_sampler_mode_value(value: Any) -> str:
    """Return a valid sampler mode for linked Generate Song/Advanced controls."""

    text = str(value or "").strip().lower()
    return text if text in {"euler", "heun"} else "heun"


def _register_simple_enhance_handlers(
    *,
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
    llm_handler: Any,
) -> None:
    """Wire simple-tab caption and lyrics enhancement buttons."""

    shared_inputs = _simple_enhance_inputs(simple_page, generation_section)
    shared_outputs = _simple_enhance_outputs(simple_page, results_section)

    simple_page["simple_enhance_caption_btn"].click(
        fn=lambda *args: _enhance_simple_caption(llm_handler, *args),
        inputs=shared_inputs,
        outputs=[
            simple_page["simple_caption"],
            *shared_outputs,
        ],
        show_progress_on=[simple_page["simple_status"]],
    )
    simple_page["simple_enhance_lyrics_btn"].click(
        fn=lambda *args: _enhance_simple_lyrics(llm_handler, *args),
        inputs=shared_inputs,
        outputs=[
            simple_page["simple_lyrics"],
            *shared_outputs,
        ],
        show_progress_on=[simple_page["simple_status"]],
    )


def _simple_enhance_inputs(
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
) -> list[Any]:
    """Return shared inputs for simple-tab LM enhancement actions."""

    return [
        simple_page["simple_caption"],
        simple_page["simple_lyrics"],
        simple_page["simple_bpm_state"],
        simple_page["simple_duration"],
        simple_page["simple_key_scale_state"],
        simple_page["simple_time_signature_state"],
        generation_section["lm_temperature"],
        generation_section["lm_top_k"],
        generation_section["lm_top_p"],
        generation_section["constrained_decoding_debug"],
        generation_section["lm_model_path"],
        generation_section["backend_dropdown"],
        generation_section["device"],
        generation_section["offload_to_cpu_checkbox"],
    ]


def _simple_enhance_outputs(
    simple_page: dict[str, Any],
    results_section: dict[str, Any],
) -> list[Any]:
    """Return shared outputs after the enhanced text field."""

    return [
        simple_page["simple_bpm_state"],
        simple_page["simple_duration"],
        simple_page["simple_key_scale_state"],
        simple_page["simple_vocal_language"],
        simple_page["simple_time_signature_state"],
        simple_page["simple_is_format_caption_state"],
        simple_page["simple_status"],
    ]


def _enhance_simple_caption(llm_handler: Any, *args: Any) -> tuple[Any, ...]:
    """Enhance the simple style/caption field with the existing LM formatter."""

    result = gen_h.handle_format_caption(llm_handler, *args)
    return _format_simple_enhance_result(result, "Caption")


def _enhance_simple_lyrics(llm_handler: Any, *args: Any) -> tuple[Any, ...]:
    """Enhance the simple lyrics field with the existing LM formatter."""

    result = gen_h.handle_format_lyrics(llm_handler, *args)
    return _format_simple_enhance_result(result, "Lyrics")


def _format_simple_enhance_result(result: Any, label: str) -> tuple[Any, ...]:
    """Make LM enhancement status compact for the simple tab."""

    if not isinstance(result, (list, tuple)) or len(result) < 8:
        return (*([gr.update()] * 7), f"{label} enhancement failed.")

    values = list(result)
    values[-1] = _format_enhancement_status(values[-1], label)
    return tuple(values)


def _format_enhancement_status(status: Any, label: str) -> str:
    """Return a concise user-facing status for simple-tab enhancement."""

    raw_status = str(status or "").strip()
    normalized = raw_status.lower()
    if "error" in normalized or "failed" in normalized or "not initialized" in normalized:
        return _compact_status(f"{label} enhancement failed.", raw_status)
    return _compact_status(f"{label} enhanced. Review before generating.", raw_status)


def _apply_simple_tier_change(tier: str | None, llm_handler: Any) -> tuple[Any, ...]:
    """Apply a GPU tier preset to advanced and simple controls."""

    updates = gen_h.on_tier_change(tier, llm_handler)
    if len(updates) != 11:
        return (gr.update(value=tier),) + tuple(gr.update() for _ in range(15))
    quantization_update = _clone_update(updates[3])
    simple_quantization_update = _clone_update(updates[3])
    batch_update = _clone_update(updates[7])
    simple_batch_update = _clone_update(updates[7])
    duration_update = _clone_update(updates[8])
    simple_duration_update = _clone_update(updates[8])
    return (
        gr.update(value=tier),
        updates[0],
        updates[1],
        updates[2],
        quantization_update,
        simple_quantization_update,
        updates[4],
        updates[5],
        updates[6],
        batch_update,
        duration_update,
        updates[9],
        simple_batch_update,
        simple_duration_update,
        updates[10],
        f"Applied GPU preset: {tier}",
    )


def _clone_update(update: Any) -> Any:
    """Return a separate update payload when one update feeds multiple components."""

    if isinstance(update, dict):
        return dict(update)
    return update


def _apply_simple_model_change(
    model_path: str | None,
    current_mode: str | None = None,
) -> tuple[Any, ...]:
    """Apply the Create-tab XL model selector to the advanced model controls."""

    selected_model = normalize_simple_model_dropdown_value(model_path)
    model_updates = list(gen_h.update_model_type_settings(selected_model, current_mode))
    quality_defaults = model_quality_defaults(selected_model)
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
    label = _SIMPLE_MODEL_LABELS.get(selected_model, selected_model)
    selected_model_lower = selected_model.lower()
    if "turbo" in selected_model_lower:
        status = (
            f"Selected model: {label}. Next generation uses XL Turbo "
            "8-step LM-conditioned defaults. GPU presets remain the XL 4B profile."
        )
    elif "base" in selected_model_lower:
        status = (
            f"Selected model: {label}. Next generation uses XL Base "
            "64-step direct DiT APG/CFG quality defaults with shift 3.0 "
            "and all task modes available. "
            "GPU presets remain the XL 4B profile."
        )
    else:
        status = (
            f"Selected model: {label}. Next generation uses XL SFT "
            "50-step CFG quality defaults with 5Hz LM Thinking metadata and shift 3.0. "
            "GPU presets remain the XL 4B profile."
        )
    return (
        gr.update(value=selected_model),
        *model_updates,
        *behavior_updates,
        *build_advanced_model_reset_updates(selected_model),
        status,
    )


def _extract_generation_status(outputs: Any) -> str:
    """Return the current generation status from a streamed output tuple."""

    if isinstance(outputs, (list, tuple)) and len(outputs) > _STATUS_OUTPUT_INDEX:
        return _format_simple_status(outputs[_STATUS_OUTPUT_INDEX])
    return ""


def _format_simple_status(status: Any) -> str:
    """Convert verbose backend status into compact simple-tab text."""

    raw_status = str(status or "").strip()
    if not raw_status:
        return "Generating song..."

    normalized = raw_status.lower()
    if "cancelled" in normalized or "canceled" in normalized:
        return "Generation cancelled."
    if "initializing dit service" in normalized:
        return _compact_status("Loading DiT model...", raw_status)
    if "initializing 5hz lm" in normalized or "initializing 5hz language" in normalized:
        return _compact_status("Loading 5Hz language model...", raw_status)
    if "preparing generation" in normalized:
        return "Preparing generated audio files..."
    if "encoding & ready" in normalized:
        return raw_status.replace("Encoding & Ready", "Encoding audio")
    if "generation complete" in normalized:
        return "Generation complete. Outputs are saved."
    if "failed" in normalized or "error" in normalized:
        return _compact_status("Generation failed.", raw_status)
    return _limit_status_lines(raw_status)


def _compact_status(title: str, details: str) -> str:
    """Return a short title plus the most useful backend detail line."""

    detail_lines = [line.strip() for line in details.splitlines() if line.strip()]
    if not detail_lines:
        return title
    last_detail = detail_lines[-1]
    if last_detail == title:
        return title
    return _limit_status_lines(f"{title}\n{last_detail}")


def _limit_status_lines(status: str, max_lines: int = 4) -> str:
    """Keep status text readable in the compact simple-tab box."""

    lines = [line.strip() for line in status.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join([*lines[: max_lines - 1], lines[-1]])
