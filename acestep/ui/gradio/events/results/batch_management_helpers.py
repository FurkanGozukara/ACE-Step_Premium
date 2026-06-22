"""Helper utilities for batch-management generation flows.

This module contains pure helper functions used by the batch wrapper and
background generation paths.
"""

from loguru import logger

from acestep.core.generation.handler.lora.folder_scan import (
    resolve_loadable_lora_adapter_path,
)
from acestep.ui.gradio.events.generation.audio_format_options import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_EXTRACT_AUDIO_FORMAT,
    DEFAULT_MP3_BITRATE,
)
from acestep.ui.gradio.events.generation.strength_defaults import (
    DEFAULT_AUDIO_COVER_STRENGTH,
)
from acestep.ui.gradio.events.generation.remix_presets import (
    REMIX_PRESET_DEFAULT,
    remix_preset_values,
)
from acestep.ui.gradio.events.dcw_defaults import get_dcw_defaults_for_think
from acestep.ui.gradio.events.results.result_output_contract import (
    AUDIO_SLOT_COUNT,
    CORE_OUTPUT_COUNT,
    SCORES_START_INDEX,
)


def _extract_ui_core_outputs(result_tuple):
    """Return the fixed core UI outputs from a generation result tuple.

    The generate-button wiring expects generation core outputs from the wrapper,
    followed by 9 batch-state outputs. Any trailing fields from
    ``generate_with_progress`` are intentionally ignored here.
    """
    return (
        tuple(result_tuple[:CORE_OUTPUT_COUNT])
        if len(result_tuple) >= CORE_OUTPUT_COUNT
        else tuple(result_tuple)
    )


def resolve_effective_lora_path(
    lora_path: str | None = None,
    lora_dropdown: str | None = None,
) -> str:
    """Return a loadable LoRA path from manual input or dropdown selection."""

    candidate = str(lora_path or "").strip() or str(lora_dropdown or "").strip()
    if not candidate:
        return ""
    return resolve_loadable_lora_adapter_path(candidate)


def apply_lora_selection_for_generation(
    dit_handler,
    lora_path: str | None = None,
    lora_dropdown: str | None = None,
    lora_scale: float | None = 1.0,
) -> tuple[str, bool, str]:
    """Synchronize the loaded LoRA state with the current UI selection.

    Returns:
        ``(resolved_path, use_lora, status_message)``.
    """

    if dit_handler is None:
        return "", False, "No LoRA will be used."

    resolved_path = resolve_effective_lora_path(lora_path, lora_dropdown)
    requested = str(lora_path or "").strip() or str(lora_dropdown or "").strip()

    if not resolved_path:
        if getattr(dit_handler, "lora_loaded", False):
            unload_status = dit_handler.unload_lora()
            if "failed" in str(unload_status).lower() or "cannot" in str(unload_status).lower():
                raise RuntimeError(f"Failed to remove LoRA before generation: {unload_status}")
        setattr(dit_handler, "_auto_lora_path", "")
        suffix = f" Invalid LoRA path: {requested}" if requested else ""
        return "", False, f"No LoRA will be used.{suffix}"

    current_auto_path = str(getattr(dit_handler, "_auto_lora_path", "") or "")
    if getattr(dit_handler, "lora_loaded", False) and current_auto_path != resolved_path:
        unload_status = dit_handler.unload_lora()
        if "failed" in str(unload_status).lower() or "cannot" in str(unload_status).lower():
            raise RuntimeError(f"Failed to switch LoRA before generation: {unload_status}")

    if not getattr(dit_handler, "lora_loaded", False):
        load_status = dit_handler.load_lora(resolved_path)
        load_status_l = str(load_status).lower()
        if any(marker in load_status_l for marker in ("failed", "invalid", "not found", "not initialized", "not supported")):
            raise RuntimeError(f"Failed to load LoRA before generation: {load_status}")
        setattr(dit_handler, "_auto_lora_path", resolved_path)

    try:
        scale_value = float(lora_scale if lora_scale is not None else 1.0)
    except (TypeError, ValueError):
        scale_value = 1.0
    if hasattr(dit_handler, "set_lora_scale"):
        dit_handler.set_lora_scale(scale_value)
    if hasattr(dit_handler, "set_use_lora"):
        dit_handler.set_use_lora(True)
    return resolved_path, True, f"Next run will use LoRA: {resolved_path}"


def _build_saved_params(
    captions, lyrics, bpm, key_scale, time_signature, vocal_language,
    inference_steps, guidance_scale, random_seed_checkbox, seed,
    reference_audio, audio_duration, batch_size_input, src_audio,
    text2music_audio_code_string, repainting_start, repainting_end,
    instruction_display_gen, audio_cover_strength, cover_noise_strength, task_type,
    no_fsq, use_adg, cfg_interval_start, cfg_interval_end, shift, infer_method,
    sampler_mode, velocity_norm_threshold, velocity_ema_factor,
    dcw_enabled, dcw_mode, dcw_scaler, dcw_high_scaler, dcw_wavelet,
    audio_format, mp3_bitrate, mp3_sample_rate, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language,
    constrained_decoding_debug, allow_lm_batch, auto_score, auto_lrc,
    score_scale, lm_batch_chunk_size,
    track_name, extract_all_stems, complete_track_classes,
    enable_normalization, normalization_db, fade_in_duration, fade_out_duration,
    latent_shift, latent_rescale,
    repaint_mode="balanced", repaint_strength=0.5,
    retake_variance=0.0, retake_seed="",
    flow_edit_morph=False,
    flow_edit_source_caption="",
    flow_edit_source_lyrics="",
    flow_edit_n_min=0.0,
    flow_edit_n_max=1.0,
    flow_edit_n_avg=1,
    lora_path="",
    lora_dropdown="",
    lora_scale=1.0,
    use_lora=False,
    generate_lm_audio_codes=None,
    extract_trim_empty_output=False,
    extract_trim_threshold_db=-40.0,
    extract_output_format=DEFAULT_EXTRACT_AUDIO_FORMAT,
    ui_runtime_settings=None,
    instrumental_checkbox=False,
    repaint_dont_switch_with_lyrics=False,
    audio_processing_settings=None,
    sam_audio_settings=None,
):
    """Build the parameter snapshot dict stored in batch history."""
    return {
        "captions": captions, "lyrics": lyrics, "bpm": bpm,
        "key_scale": key_scale, "time_signature": time_signature,
        "vocal_language": vocal_language, "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "random_seed_checkbox": random_seed_checkbox, "seed": seed,
        "reference_audio": reference_audio, "audio_duration": audio_duration,
        "batch_size_input": batch_size_input, "src_audio": src_audio,
        "text2music_audio_code_string": text2music_audio_code_string,
        "repainting_start": repainting_start, "repainting_end": repainting_end,
        "instruction_display_gen": instruction_display_gen,
        "audio_cover_strength": audio_cover_strength,
        "cover_noise_strength": cover_noise_strength,
        "task_type": task_type, "no_fsq": no_fsq, "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift, "infer_method": infer_method,
        "sampler_mode": sampler_mode,
        "velocity_norm_threshold": velocity_norm_threshold,
        "velocity_ema_factor": velocity_ema_factor,
        "dcw_enabled": dcw_enabled,
        "dcw_mode": dcw_mode,
        "dcw_scaler": dcw_scaler,
        "dcw_high_scaler": dcw_high_scaler,
        "dcw_wavelet": dcw_wavelet,
        "audio_format": audio_format,
        "mp3_bitrate": mp3_bitrate,
        "mp3_sample_rate": mp3_sample_rate,
        "lm_temperature": lm_temperature,
        "think_checkbox": think_checkbox, "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k, "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas, "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score, "auto_lrc": auto_lrc,
        "score_scale": score_scale, "lm_batch_chunk_size": lm_batch_chunk_size,
        "track_name": track_name,
        "extract_all_stems": bool(extract_all_stems),
        "complete_track_classes": complete_track_classes,
        "enable_normalization": enable_normalization,
        "normalization_db": normalization_db,
        "fade_in_duration": fade_in_duration,
        "fade_out_duration": fade_out_duration,
        "latent_shift": latent_shift, "latent_rescale": latent_rescale,
        "repaint_mode": repaint_mode, "repaint_strength": repaint_strength,
        "retake_variance": retake_variance, "retake_seed": retake_seed,
        "flow_edit_morph": flow_edit_morph,
        "flow_edit_source_caption": flow_edit_source_caption,
        "flow_edit_source_lyrics": flow_edit_source_lyrics,
        "flow_edit_n_min": flow_edit_n_min,
        "flow_edit_n_max": flow_edit_n_max,
        "flow_edit_n_avg": flow_edit_n_avg,
        "lora_path": lora_path,
        "lora_dropdown": lora_dropdown,
        "lora_scale": lora_scale,
        "use_lora": use_lora,
        "generate_lm_audio_codes": generate_lm_audio_codes,
        "extract_trim_empty_output": extract_trim_empty_output,
        "extract_trim_threshold_db": extract_trim_threshold_db,
        "extract_output_format": extract_output_format,
        "ui_runtime_settings": ui_runtime_settings or {},
        "instrumental_checkbox": bool(instrumental_checkbox),
        "repaint_dont_switch_with_lyrics": bool(repaint_dont_switch_with_lyrics),
        "audio_processing_settings": audio_processing_settings or {},
        "sam_audio_settings": sam_audio_settings or {},
    }


def _log_background_params(params, next_batch_idx):
    """Log background-generation parameter values for diagnostics."""
    logger.info(f"========== BACKGROUND GENERATION BATCH {next_batch_idx + 1} ==========")
    logger.info(f"  - captions: {params.get('captions', 'N/A')}")
    lyr = params.get("lyrics")
    logger.info(f"  - lyrics: {lyr[:50]}..." if lyr else "  - lyrics: N/A")
    logger.info(f"  - bpm: {params.get('bpm')}")
    logger.info(f"  - config_path: {params.get('config_path', 'active foreground DiT service')}")
    logger.info(f"  - inference_steps: {params.get('inference_steps')}")
    logger.info(f"  - songs: {params.get('batch_size_input')}")
    logger.info(f"  - allow_lm_batch: {params.get('allow_lm_batch')}")
    logger.info(f"  - think_checkbox: {params.get('think_checkbox')}")
    logger.info(f"  - no_fsq: {params.get('no_fsq')}")
    logger.info(f"  - lm_temperature: {params.get('lm_temperature')}")
    logger.info(f"  - track_name: {params.get('track_name')}")
    logger.info(f"  - extract_all_stems: {params.get('extract_all_stems')}")
    codes_val = params.get("text2music_audio_code_string")
    logger.info(f"  - text2music_audio_code_string: {'<CLEARED>' if codes_val == '' else 'HAS_VALUE'}")
    logger.info("=========================================================")


def _apply_param_defaults(params):
    """Fill missing generation keys in ``params`` with safe defaults."""
    dcw_defaults = get_dcw_defaults_for_think(bool(params.get("think_checkbox", True)))
    task_type = str(params.get("task_type") or "text2music")
    remix_strength_default, remix_retention_default = remix_preset_values(
        REMIX_PRESET_DEFAULT
    )
    audio_cover_default = (
        remix_strength_default
        if task_type in ("cover", "cover-nofsq")
        else DEFAULT_AUDIO_COVER_STRENGTH
    )
    cover_noise_default = (
        remix_retention_default
        if task_type in ("cover", "cover-nofsq")
        else 0.0
    )
    defaults = {
        "captions": "", "lyrics": "", "bpm": None, "key_scale": "",
        "time_signature": "", "vocal_language": "unknown",
        "inference_steps": 8, "guidance_scale": 7.0,
        "random_seed_checkbox": True, "seed": "-1",
        "reference_audio": None, "audio_duration": -1,
        "batch_size_input": 1, "src_audio": None,
        "text2music_audio_code_string": "",
        "repainting_start": 0.0, "repainting_end": -1,
        "instruction_display_gen": "",
        "audio_cover_strength": audio_cover_default,
        "cover_noise_strength": cover_noise_default,
        "task_type": task_type, "no_fsq": False, "use_adg": False,
        "cfg_interval_start": 0.0, "cfg_interval_end": 1.0,
        "shift": 1.0, "infer_method": "ode",
        "sampler_mode": "heun", "velocity_norm_threshold": 0.0,
        "velocity_ema_factor": 0.0,
        "dcw_enabled": True,
        "dcw_mode": dcw_defaults["mode"],
        "dcw_scaler": dcw_defaults["scaler"],
        "dcw_high_scaler": dcw_defaults["high_scaler"],
        "dcw_wavelet": "haar",
        "custom_timesteps": "",
        "audio_format": DEFAULT_AUDIO_FORMAT,
        "mp3_bitrate": DEFAULT_MP3_BITRATE,
        "mp3_sample_rate": 48000,
        "lm_temperature": 0.85,
        "think_checkbox": True, "lm_cfg_scale": 2.0,
        "generate_lm_audio_codes": None,
        "lm_top_k": 0, "lm_top_p": 0.9,
        "lm_negative_prompt": "",
        "use_cot_metas": True, "use_cot_caption": False,
        "use_cot_language": False,
        "constrained_decoding_debug": False,
        "allow_lm_batch": True, "auto_score": False,
        "auto_lrc": False, "score_scale": 0.5,
        "lm_batch_chunk_size": 8,
        "track_name": None, "extract_all_stems": False,
        "complete_track_classes": [],
        "enable_normalization": True, "normalization_db": -1.0,
        "fade_in_duration": 0.0, "fade_out_duration": 0.0,
        "extract_trim_empty_output": False,
        "extract_trim_threshold_db": -40.0,
        "extract_output_format": DEFAULT_EXTRACT_AUDIO_FORMAT,
        "instrumental_checkbox": False,
        "repaint_dont_switch_with_lyrics": False,
        "latent_shift": 0.0, "latent_rescale": 1.0,
        "repaint_mode": "balanced", "repaint_strength": 0.5,
        "retake_variance": 0.0, "retake_seed": "",
        "flow_edit_morph": False,
        "flow_edit_source_caption": "",
        "flow_edit_source_lyrics": "",
        "flow_edit_n_min": 0.0,
        "flow_edit_n_max": 1.0,
        "flow_edit_n_avg": 1,
        "lora_path": "",
        "lora_dropdown": "",
        "lora_scale": 1.0,
        "use_lora": False,
        "ui_runtime_settings": {},
        "audio_processing_settings": {},
        "sam_audio_settings": {},
    }
    for key, value in defaults.items():
        if key not in params or params.get(key) is None:
            params[key] = value


def _extract_scores(final_result):
    """Extract score strings from the generation result tuple."""
    scores = []
    for idx in range(SCORES_START_INDEX, SCORES_START_INDEX + AUDIO_SLOT_COUNT):
        if idx < len(final_result):
            val = final_result[idx]
            if hasattr(val, "value"):
                scores.append(val.value if val.value else "")
            elif isinstance(val, str):
                scores.append(val)
            else:
                scores.append("")
        else:
            scores.append("")
    return scores
