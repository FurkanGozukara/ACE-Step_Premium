"""Parameter mapping for the simple Gradio Create tab."""

from __future__ import annotations

import re
from typing import Any


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
    formatted_bpm: Any = None,
    formatted_key_scale: Any = "",
    formatted_time_signature: Any = "",
    is_format_caption: bool | None = False,
) -> tuple[Any, ...]:
    """Map simple Create inputs onto the full generation component contract."""

    final_caption = _apply_vocal_direction(caption or "", instrumental, vocal_gender)
    final_lyrics = "" if instrumental else (lyrics or "")
    final_language = "unknown" if instrumental else (vocal_language or "unknown")
    duration_value = _normalize_duration(duration)
    auto_duration = _is_auto_duration(duration_value)
    batch_value = max(1, int(batch_size or 1))
    random_seed_value = bool(random_seed)
    seed_value = _normalize_seed(seed)
    bpm_value = _normalize_bpm(formatted_bpm)
    key_scale_value = _normalize_text_value(formatted_key_scale)
    time_signature_value = _normalize_text_value(formatted_time_signature)
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
        auto_duration,
        auto_duration,
        random_seed_value,
        seed_value,
        "",
        bpm_value is None,
        not key_scale_value,
        not time_signature_value,
        True,
        auto_duration,
        auto_duration,
        auto_duration,
        auto_duration,
        bpm_value,
        key_scale_value,
        time_signature_value,
        bool(is_format_caption),
        status,
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


def _is_auto_duration(duration: float) -> bool:
    """Return whether the value requests LM-estimated automatic duration."""

    return duration <= 0


def _build_status(auto_duration: bool, duration: float) -> str:
    """Build the simple-tab generation status message."""

    if auto_duration:
        return (
            "Generation started. Auto duration will be estimated from the "
            "prompt and lyrics with the 5Hz LM."
        )
    return f"Generation started. Fixed duration: {duration:g}s."
