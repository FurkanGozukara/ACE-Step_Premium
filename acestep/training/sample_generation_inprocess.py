"""In-process LoRA training sample generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from acestep.training.sample_generation_runtime import (
    cuda_peak_gb,
    release_memory,
    sample_runtime_context,
    serializable_audios,
)


def run_training_sample_inprocess(
    *,
    handler: Any,
    output_dir: str,
    prompt: str,
    lyrics: str,
    generation_settings: dict[str, Any],
    fallback_duration: float,
    fallback_inference_steps: int,
    fallback_seed: int,
    offload_generation: bool,
) -> dict[str, Any]:
    """Generate a checkpoint sample with the already-loaded training handler."""

    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.ui.gradio.events.generation.audio_format_options import (
        normalize_audio_format,
        primary_audio_format,
    )
    from acestep.ui.gradio.events.generation.validation import (
        parse_and_validate_timesteps,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "sample_result.json"
    settings = dict(generation_settings or {})
    inference_steps = _as_int(settings.get("inference_steps"), fallback_inference_steps)
    timesteps, _warned, _message = parse_and_validate_timesteps(
        str(settings.get("custom_timesteps") or ""),
        inference_steps,
    )
    if timesteps is not None:
        inference_steps = max(1, len(timesteps) - 1)

    audio_format = normalize_audio_format(settings.get("audio_format"))
    backend_audio_format = primary_audio_format(audio_format)
    seed = _resolve_seed(settings, fallback_seed)
    duration = _as_float(settings.get("audio_duration"), fallback_duration)

    params = GenerationParams(
        task_type="text2music",
        thinking=False,
        caption=prompt,
        lyrics=lyrics,
        instrumental="[instrumental]" in str(lyrics).lower(),
        vocal_language=str(settings.get("vocal_language") or "en"),
        bpm=_as_optional_int(settings.get("bpm")),
        keyscale=str(settings.get("key_scale") or ""),
        timesignature=str(settings.get("time_signature") or ""),
        duration=duration,
        inference_steps=inference_steps,
        guidance_scale=_as_float(settings.get("guidance_scale"), 1.0),
        use_adg=bool(settings.get("use_adg", False)),
        cfg_interval_start=_as_float(settings.get("cfg_interval_start"), 0.0),
        cfg_interval_end=_as_float(settings.get("cfg_interval_end"), 1.0),
        shift=_as_float(settings.get("shift"), 3.0),
        infer_method=str(settings.get("infer_method") or "ode"),
        sampler_mode=str(settings.get("sampler_mode") or "euler"),
        velocity_norm_threshold=_as_float(
            settings.get("velocity_norm_threshold"),
            0.0,
        ),
        velocity_ema_factor=_as_float(settings.get("velocity_ema_factor"), 0.0),
        dcw_enabled=bool(settings.get("dcw_enabled", True)),
        dcw_mode=str(settings.get("dcw_mode") or "double"),
        dcw_scaler=_as_float(settings.get("dcw_scaler"), 0.05),
        dcw_high_scaler=_as_float(settings.get("dcw_high_scaler"), 0.02),
        dcw_wavelet=str(settings.get("dcw_wavelet") or "haar"),
        timesteps=timesteps,
        seed=seed,
        enable_normalization=bool(settings.get("enable_normalization", True)),
        normalization_db=_as_float(settings.get("normalization_db"), -1.0),
        fade_in_duration=_as_float(settings.get("fade_in_duration"), 0.0),
        fade_out_duration=_as_float(settings.get("fade_out_duration"), 0.0),
        latent_shift=_as_float(settings.get("latent_shift"), 0.0),
        latent_rescale=_as_float(settings.get("latent_rescale"), 1.0),
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_constrained_decoding=bool(
            settings.get("constrained_decoding_debug") is not None
        ),
    )
    config = GenerationConfig(
        batch_size=1,
        allow_lm_batch=False,
        use_random_seed=bool(settings.get("random_seed_checkbox", False)),
        seeds=[seed],
        constrained_decoding_debug=bool(
            settings.get("constrained_decoding_debug", False)
        ),
        audio_format=backend_audio_format,
        mp3_bitrate=str(settings.get("mp3_bitrate") or "320k"),
        mp3_sample_rate=_as_int(settings.get("mp3_sample_rate"), 48000),
    )

    peak_before = cuda_peak_gb()
    with sample_runtime_context(handler, enabled=offload_generation):
        generated = generate_music(
            handler,
            None,
            params=params,
            config=config,
            save_dir=str(output_path),
        )
    payload = {
        "success": bool(generated.success),
        "error": generated.error or generated.status_message,
        "audios": serializable_audios(generated.audios),
        "peak_vram_gb": cuda_peak_gb(),
        "peak_vram_before_gb": peak_before,
        "audio_format": backend_audio_format,
        "inference_steps": inference_steps,
        "duration": duration,
    }
    payload["peak_vram_increase_gb"] = max(
        0.0,
        payload["peak_vram_gb"] - payload["peak_vram_before_gb"],
    )
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    release_memory()
    return payload


def _resolve_seed(settings: dict[str, Any], fallback: int) -> int:
    """Resolve the seed to pass into sample generation."""

    if bool(settings.get("random_seed_checkbox", False)):
        return -1
    return _as_int(settings.get("seed"), fallback)


def _as_int(value: Any, default: int) -> int:
    """Convert a UI value to int."""

    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _as_optional_int(value: Any) -> int | None:
    """Convert a UI metadata value to an optional int."""

    parsed = _as_int(value, 0)
    return parsed if parsed > 0 else None


def _as_float(value: Any, default: float) -> float:
    """Convert a UI value to float."""

    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)

