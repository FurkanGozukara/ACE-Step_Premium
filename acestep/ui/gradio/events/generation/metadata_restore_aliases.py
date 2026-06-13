"""Saved-key alias applicators for Load Metadata restore values."""

from __future__ import annotations

from typing import Any

from .audio_format_options import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_EXTRACT_AUDIO_FORMAT,
    DEFAULT_MP3_BITRATE,
    normalize_audio_format,
    normalize_extract_audio_format,
)
from .metadata_restore_document import first_value


def apply_simple_aliases(
    values: dict[str, Any],
    payload: dict[str, Any],
    generation_params: dict[str, Any],
) -> None:
    """Apply direct key aliases from metadata to UI values."""

    aliases = {
        "use_adg": ("use_adg",),
        "cfg_interval_start": ("cfg_interval_start",),
        "cfg_interval_end": ("cfg_interval_end",),
        "shift": ("shift",),
        "infer_method": ("infer_method",),
        "sampler_mode": ("sampler_mode",),
        "velocity_norm_threshold": ("velocity_norm_threshold",),
        "velocity_ema_factor": ("velocity_ema_factor",),
        "dcw_enabled": ("dcw_enabled",),
        "dcw_mode": ("dcw_mode",),
        "dcw_scaler": ("dcw_scaler",),
        "dcw_high_scaler": ("dcw_high_scaler",),
        "dcw_wavelet": ("dcw_wavelet",),
        "custom_timesteps": ("custom_timesteps", "timesteps"),
        "lm_temperature": ("lm_temperature",),
        "lm_cfg_scale": ("lm_cfg_scale",),
        "lm_top_k": ("lm_top_k",),
        "lm_top_p": ("lm_top_p",),
        "lm_negative_prompt": ("lm_negative_prompt",),
        "use_cot_metas": ("use_cot_metas",),
        "use_cot_caption": ("use_cot_caption",),
        "use_cot_language": ("use_cot_language",),
        "audio_cover_strength": ("audio_cover_strength",),
        "cover_noise_strength": ("cover_noise_strength",),
        "think_checkbox": ("thinking", "think_checkbox"),
        "text2music_audio_code_string": ("text2music_audio_code_string", "audio_codes"),
        "repainting_start": ("repainting_start",),
        "repainting_end": ("repainting_end",),
        "complete_track_classes": ("complete_track_classes",),
        "instrumental_checkbox": ("instrumental_checkbox", "instrumental"),
        "repaint_dont_switch_with_lyrics": ("repaint_dont_switch_with_lyrics",),
        "retake_variance": ("retake_variance",),
        "retake_seed": ("retake_seed", "retake_seed_value"),
        "flow_edit_morph": ("flow_edit_morph",),
        "flow_edit_source_caption": ("flow_edit_source_caption",),
        "flow_edit_source_lyrics": ("flow_edit_source_lyrics",),
        "flow_edit_n_min": ("flow_edit_n_min",),
        "flow_edit_n_max": ("flow_edit_n_max",),
        "flow_edit_n_avg": ("flow_edit_n_avg",),
        "enable_normalization": ("enable_normalization",),
        "normalization_db": ("normalization_db",),
        "fade_in_duration": ("fade_in_duration",),
        "fade_out_duration": ("fade_out_duration",),
        "latent_shift": ("latent_shift",),
        "latent_rescale": ("latent_rescale",),
        "no_fsq": ("no_fsq",),
        "constrained_decoding_debug": ("constrained_decoding_debug",),
        "allow_lm_batch": ("allow_lm_batch",),
        "auto_score": ("auto_score",),
        "auto_lrc": ("auto_lrc",),
        "score_scale": ("score_scale",),
        "lm_batch_chunk_size": ("lm_batch_chunk_size",),
        "generate_lm_audio_codes": ("generate_lm_audio_codes",),
        "extract_trim_empty_output": ("extract_trim_empty_output",),
        "extract_trim_threshold_db": ("extract_trim_threshold_db",),
    }
    for output_key, source_keys in aliases.items():
        value = first_value(payload, generation_params, *source_keys)
        if value is not None:
            values[output_key] = value


def apply_audio_format_values(
    values: dict[str, Any],
    payload: dict[str, Any],
    task_type: Any,
) -> None:
    """Normalize generated-audio and MP3 control values."""

    values["audio_format"] = normalize_audio_format(
        first_value(payload, "audio_format", default=DEFAULT_AUDIO_FORMAT)
    )
    mp3_bitrate = str(
        first_value(payload, "mp3_bitrate", default=DEFAULT_MP3_BITRATE)
    ).strip().lower()
    values["mp3_bitrate"] = (
        mp3_bitrate
        if mp3_bitrate in {"128k", "192k", "256k", "320k"}
        else DEFAULT_MP3_BITRATE
    )
    try:
        mp3_sample_rate = int(first_value(payload, "mp3_sample_rate", default=48000))
    except (TypeError, ValueError):
        mp3_sample_rate = 48000
    values["mp3_sample_rate"] = (
        mp3_sample_rate if mp3_sample_rate in {44100, 48000} else 48000
    )
    if task_type == "extract":
        values["extract_output_format"] = normalize_extract_audio_format(
            values["audio_format"]
        )


def apply_extract_values(
    values: dict[str, Any],
    payload: dict[str, Any],
    generation_params: dict[str, Any],
    task_type: Any,
) -> None:
    """Apply Extract and Complete track metadata values."""

    values["track_name"] = first_value(payload, generation_params, "track_name")
    values["extract_all_stems"] = bool(
        first_value(payload, generation_params, "extract_all_stems", default=False)
    )
    extract_format = first_value(
        payload,
        generation_params,
        "extract_output_format",
        default=None,
    )
    if extract_format is None and task_type == "extract":
        extract_format = values["audio_format"]
    values["extract_output_format"] = normalize_extract_audio_format(
        extract_format or DEFAULT_EXTRACT_AUDIO_FORMAT
    )


def apply_runtime_values(values: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Apply runtime/service UI fields saved in the manifest."""

    for key in (
        "config_path",
        "device",
        "vae_checkpoint",
        "lm_model_path",
        "backend_dropdown",
        "init_llm_checkbox",
        "lm_use_legacy_cfg_prompt",
        "use_flash_attention_checkbox",
        "offload_to_cpu_checkbox",
        "offload_dit_to_cpu_checkbox",
        "compile_model_checkbox",
        "quantization_checkbox",
        "mlx_dit_checkbox",
        "mlx_vae_chunk_size",
        "lora_path",
        "lora_dropdown",
        "use_lora_checkbox",
        "lora_scale_slider",
        "subprocess_mode_checkbox",
    ):
        if runtime.get(key) is not None:
            values[key] = runtime[key]
