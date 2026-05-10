"""Event wiring for the simple first-tab creation flow."""

from __future__ import annotations

from typing import Any

import gradio as gr

from .. import generation_handlers as gen_h
from .. import results_handlers as res_h
from ..generation.quantization import default_quantization_value
from ...premium_features import (
    SIMPLE_MODEL_CHOICES,
    normalize_simple_model_dropdown_value,
    open_outputs_folder,
)
from .generation_run_wiring import (
    build_clear_audio_outputs,
    build_generation_run_inputs,
    build_generation_run_outputs,
)
from .simple_media_outputs import build_simple_media_preview, clear_simple_media_preview
from .simple_create_params import prepare_simple_generation


_STATUS_OUTPUT_INDEX = 10
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

    def generation_wrapper(*args: Any):
        for outputs in res_h.generate_with_batch_management(dit_handler, llm_handler, *args):
            status = _extract_generation_status(outputs)
            yield (*outputs, status)

    simple_page["simple_open_outputs_btn"].click(
        fn=open_outputs_folder,
        inputs=[],
        outputs=[simple_page["simple_status"]],
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
            simple_page["simple_status"],
        ],
    )

    simple_page["simple_model_dropdown"].change(
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
            simple_page["simple_status"],
        ],
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
        inputs=[
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
        ],
        outputs=_simple_prepare_outputs(generation_section, results_section, simple_page),
    ).then(
        fn=res_h.clear_audio_outputs_for_new_generation,
        outputs=build_clear_audio_outputs(results_section),
    ).then(
        fn=clear_simple_media_preview,
        outputs=[
            simple_page["simple_latest_audio"],
            simple_page["simple_latest_video"],
        ],
    ).then(
        fn=generation_wrapper,
        inputs=build_generation_run_inputs(generation_section, results_section),
        outputs=[
            *build_generation_run_outputs(generation_section, results_section),
            simple_page["simple_status"],
        ],
    ).then(
        fn=build_simple_media_preview,
        inputs=[
            results_section["generated_audio_1"],
            results_section["status_output"],
            simple_page["simple_cover_image"],
            simple_page["simple_video_resolution"],
        ],
        outputs=[
            simple_page["simple_latest_audio"],
            simple_page["simple_latest_video"],
            simple_page["simple_status"],
        ],
    )


def _simple_prepare_outputs(
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
    simple_page: dict[str, Any],
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
        simple_page["simple_status"],
        generation_section["config_path"],
    ]


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
    )
    simple_page["simple_enhance_lyrics_btn"].click(
        fn=lambda *args: _enhance_simple_lyrics(llm_handler, *args),
        inputs=shared_inputs,
        outputs=[
            simple_page["simple_lyrics"],
            *shared_outputs,
        ],
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
    if len(updates) != 10:
        return (gr.update(value=tier),) + tuple(gr.update() for _ in range(14))
    quantization_update = updates[3]
    batch_update = updates[7]
    duration_update = updates[8]
    return (
        gr.update(value=tier),
        updates[0],
        updates[1],
        updates[2],
        quantization_update,
        quantization_update,
        updates[4],
        updates[5],
        updates[6],
        updates[7],
        updates[8],
        updates[9],
        batch_update,
        duration_update,
        f"Applied GPU preset: {tier}",
    )


def _apply_simple_model_change(
    model_path: str | None,
    current_mode: str | None = None,
) -> tuple[Any, ...]:
    """Apply the Create-tab SFT/Turbo selector to the advanced model controls."""

    selected_model = normalize_simple_model_dropdown_value(model_path)
    model_updates = gen_h.update_model_type_settings(selected_model, current_mode)
    label = _SIMPLE_MODEL_LABELS.get(selected_model, selected_model)
    if "turbo" in selected_model:
        status = (
            f"Selected model: {label}. Next generation uses XL Turbo "
            "8-step fast defaults. GPU presets remain the XL 4B profile."
        )
    else:
        status = (
            f"Selected model: {label}. Next generation uses XL SFT "
            "50-step quality defaults. GPU presets remain the XL 4B profile."
        )
    return (gr.update(value=selected_model), *model_updates, status)


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
