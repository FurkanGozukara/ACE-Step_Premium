"""Advanced generation-setting mapping for LoRA checkpoint samples."""

from __future__ import annotations

from typing import Any


SAMPLE_GENERATION_SETTING_KEYS = (
    "config_path",
    "audio_duration",
    "inference_steps",
    "guidance_scale",
    "random_seed_checkbox",
    "seed",
    "bpm",
    "key_scale",
    "time_signature",
    "vocal_language",
    "no_fsq",
    "use_adg",
    "cfg_interval_start",
    "cfg_interval_end",
    "shift",
    "infer_method",
    "sampler_mode",
    "velocity_norm_threshold",
    "velocity_ema_factor",
    "dcw_enabled",
    "dcw_mode",
    "dcw_scaler",
    "dcw_high_scaler",
    "dcw_wavelet",
    "custom_timesteps",
    "audio_format",
    "mp3_bitrate",
    "mp3_sample_rate",
    "enable_normalization",
    "normalization_db",
    "fade_in_duration",
    "fade_out_duration",
    "latent_shift",
    "latent_rescale",
    "constrained_decoding_debug",
)


def sample_generation_input_components(generation_section: Any) -> list[Any]:
    """Return Advanced-tab generation controls consumed by training samples."""

    if not isinstance(generation_section, dict):
        return []
    return [
        generation_section[key]
        for key in SAMPLE_GENERATION_SETTING_KEYS
        if key in generation_section
    ]


def sample_generation_setting_keys(generation_section: Any) -> tuple[str, ...]:
    """Return ordered setting keys available in the generation component map."""

    if not isinstance(generation_section, dict):
        return ()
    return tuple(
        key for key in SAMPLE_GENERATION_SETTING_KEYS if key in generation_section
    )
