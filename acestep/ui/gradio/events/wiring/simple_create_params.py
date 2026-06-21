"""Parameter mapping for the simple Gradio Create tab."""

from __future__ import annotations

import math
import re
from typing import Any

from acestep.constants import DURATION_MAX, DURATION_MIN
from ..generation.generation_count import normalize_generation_count
from ...premium_features import (
    model_quality_defaults,
    normalize_simple_model_dropdown_value,
)


def prepare_simple_generation(
    caption: str,
    lyrics: str,
    vocal_language: str,
    instrumental: bool,
    vocal_gender: str | None,
    duration: float | int | str | None,
    batch_size: float | int | None,
    random_seed: bool | None,
    seed: Any,
    quantization: str | None,
    model_path: str | None = None,
    formatted_bpm: Any = None,
    formatted_key_scale: Any = "",
    formatted_time_signature: Any = "",
    is_format_caption: bool | None = False,
    negative_prompt: str = "",
) -> tuple[Any, ...]:
    """Map simple Create inputs onto the full generation component contract."""

    final_caption = _apply_vocal_direction(caption or "", instrumental, vocal_gender)
    final_lyrics = "" if instrumental else (lyrics or "")
    final_language = "unknown" if instrumental else (vocal_language or "unknown")
    duration_value = _normalize_duration(duration)
    auto_duration = _is_auto_duration(duration_value)
    batch_value = normalize_generation_count(batch_size)
    random_seed_value = bool(random_seed)
    seed_value = _normalize_seed(seed)
    bpm_value = _normalize_bpm(formatted_bpm)
    key_scale_value = _normalize_text_value(formatted_key_scale)
    time_signature_value = _normalize_text_value(formatted_time_signature)
    selected_model = normalize_simple_model_dropdown_value(model_path)
    quality_defaults = model_quality_defaults(selected_model)
    model_uses_lm = bool(quality_defaults["think_checkbox"])
    use_lm_for_generation = model_uses_lm
    if auto_duration and not use_lm_for_generation:
        duration_value = _estimate_direct_auto_duration(final_lyrics, final_caption)
        auto_duration = False
        status = _build_status(auto_duration, duration_value, estimated=True)
    else:
        status = _build_status(auto_duration, duration_value)

    return (
        "Custom",
        "text2music",
        final_caption,
        final_lyrics,
        final_language,
        duration_value,
        batch_value,
        quantization or "none",
        use_lm_for_generation,
        use_lm_for_generation,
        quality_defaults["allow_lm_batch"] if use_lm_for_generation else False,
        random_seed_value,
        seed_value,
        "",
        bpm_value is None,
        not key_scale_value,
        not time_signature_value,
        True,
        auto_duration,
        use_lm_for_generation,
        False,
        False,
        bpm_value,
        key_scale_value,
        time_signature_value,
        bool(is_format_caption),
        status,
        quality_defaults["inference_steps"],
        quality_defaults["guidance_scale"],
        quality_defaults["use_adg"],
        quality_defaults["shift"],
        quality_defaults["cfg_interval_start"],
        quality_defaults["cfg_interval_end"],
        selected_model,
        quality_defaults["dcw_enabled"],
        quality_defaults["dcw_mode"],
        quality_defaults["dcw_scaler"],
        quality_defaults["dcw_high_scaler"],
        quality_defaults["infer_method"],
        quality_defaults["sampler_mode"],
        quality_defaults["velocity_norm_threshold"],
        quality_defaults["velocity_ema_factor"],
        quality_defaults["custom_timesteps"],
        quality_defaults["dcw_wavelet"],
        quality_defaults["generate_lm_audio_codes"],
        _normalize_negative_prompt(negative_prompt),
    )


def _normalize_seed(seed: Any) -> str:
    """Return the simple-tab seed string expected by the advanced generator."""

    text = str(seed or "").strip()
    return text or "-1"


def _apply_vocal_direction(caption: str, instrumental: bool, vocal_gender: str | None) -> str:
    """Apply simple-tab vocal intent to the ACE-Step style prompt."""

    cleaned_caption = " ".join(str(caption or "").split())
    if instrumental:
        return _append_prompt_direction(
            _remove_vocal_gender_terms(cleaned_caption),
            "instrumental arrangement, no vocals",
        )

    gender = str(vocal_gender or "").strip().lower()
    if gender not in {"male", "female"}:
        return cleaned_caption

    direction = f"{gender} vocal"
    if _has_vocal_gender_term(cleaned_caption):
        return _replace_vocal_gender_terms(cleaned_caption, direction)
    return _append_prompt_direction(cleaned_caption, direction)


def _append_prompt_direction(caption: str, direction: str) -> str:
    """Append a concise style direction without duplicating punctuation."""

    if not caption:
        return direction
    normalized = caption.rstrip(" .,;")
    return f"{normalized}, {direction}."


def _has_vocal_gender_term(caption: str) -> bool:
    """Return whether the prompt already contains a gendered vocal term."""

    return bool(re.search(r"\b(?:male|female)\s+(?:vocal|vocals|voice)\b", caption, re.I))


def _replace_vocal_gender_terms(caption: str, direction: str) -> str:
    """Replace existing gendered vocal terms with the selected direction."""

    return re.sub(
        r"\b(?:male|female)\s+(?:vocal|vocals|voice)\b",
        direction,
        caption,
        flags=re.I,
    )


def _remove_vocal_gender_terms(caption: str) -> str:
    """Remove direct gendered vocal directions for instrumental requests."""

    without_terms = re.sub(
        r"\b(?:soulful\s+|confident\s+|clean\s+|powerful\s+|warm\s+)*"
        r"(?:male|female)\s+(?:vocal|vocals|voice)\b",
        "",
        caption,
        flags=re.I,
    )
    return re.sub(r"\s*,\s*,+", ",", without_terms).strip(" ,.")


def _normalize_duration(duration: float | int | str | None) -> float:
    """Return a numeric duration, using -1 for auto/invalid values."""

    if duration in (None, ""):
        return -1.0
    try:
        return float(duration)
    except (TypeError, ValueError):
        return -1.0


def _normalize_bpm(value: Any) -> int | None:
    """Return a positive integer BPM or None when unavailable."""

    if value in (None, ""):
        return None
    try:
        bpm = int(float(value))
    except (TypeError, ValueError):
        return None
    return bpm if bpm > 0 else None


def _normalize_text_value(value: Any) -> str:
    """Return a clean metadata text value, excluding empty and N/A values."""

    text = str(value or "").strip()
    if not text or text.lower() == "n/a":
        return ""
    return text


def _normalize_negative_prompt(value: Any) -> str:
    """Return the negative prompt text; the old sentinel should behave as empty."""

    text = str(value or "").strip()
    return "" if text.upper() == "NO USER INPUT" else text


def _is_auto_duration(duration: float) -> bool:
    """Return whether the value requests automatic duration."""

    return duration <= 0


def _estimate_direct_auto_duration(lyrics: str, caption: str) -> float:
    """Estimate a fixed duration for non-LM Base/SFT generation."""

    lyric_text = str(lyrics or "")
    without_tags = re.sub(r"\[[^\]]+\]", " ", lyric_text)
    words = re.findall(r"[A-Za-z0-9']+", without_tags)
    if len(words) < 80:
        return 60.0

    context = " ".join([str(caption or ""), lyric_text]).lower()
    rap_like = bool(
        re.search(
            r"\b(rap|hip[-\s]?hop|trap|drill|grime|pop[-\s]?rap|808|boom[-\s]?bap|r&b|rnb)\b",
            context,
        )
    )
    max_words_per_second = 2.35 if rap_like else 2.10
    seconds = int(math.ceil(len(words) / max_words_per_second))
    return float(max(DURATION_MIN, min(DURATION_MAX, seconds)))


def _build_status(auto_duration: bool, duration: float, estimated: bool = False) -> str:
    """Build the simple-tab generation status message."""

    if auto_duration:
        return (
            "Generation started. Auto duration will be estimated from the "
            "prompt and lyrics with the 5Hz LM."
        )
    if estimated:
        return f"Generation started. Auto duration estimated from lyrics: {duration:g}s."
    return f"Generation started. Fixed duration: {duration:g}s."
