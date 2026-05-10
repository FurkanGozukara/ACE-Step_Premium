"""Generation-argument helpers for batch folder processing."""

from __future__ import annotations

from typing import Any, Sequence


CAPTION_ARG_INDEX = 0
LYRICS_ARG_INDEX = 1
BPM_ARG_INDEX = 2
KEY_SCALE_ARG_INDEX = 3
TIME_SIGNATURE_ARG_INDEX = 4
AUDIO_DURATION_ARG_INDEX = 11
IS_FORMAT_CAPTION_ARG_INDEX = 48
AUTOGEN_ARG_INDEX = 73
CURRENT_BATCH_INDEX_ARG_INDEX = 74
TOTAL_BATCHES_ARG_INDEX = 75
BATCH_QUEUE_ARG_INDEX = 76
GENERATION_PARAMS_STATE_ARG_INDEX = 77
LM_TEMPERATURE_ARG_INDEX = 39
LM_TOP_K_ARG_INDEX = 42
LM_TOP_P_ARG_INDEX = 43
DEVICE_ARG_INDEX = 80
LM_MODEL_PATH_ARG_INDEX = 81
BACKEND_ARG_INDEX = 82
OFFLOAD_TO_CPU_ARG_INDEX = 85
LORA_SCALE_ARG_INDEX = 92
GENERATION_ARG_COUNT = LORA_SCALE_ARG_INDEX + 1


def build_generation_args_for_item(
    generation_args: Sequence[Any],
    *,
    caption: str,
    lyrics: str,
    is_formatted: bool,
) -> list[Any]:
    """Return generation args with item-specific caption and lyrics inserted."""

    args = list(generation_args)
    if len(args) < GENERATION_ARG_COUNT:
        raise ValueError(
            f"Expected at least {GENERATION_ARG_COUNT} generation settings, got {len(args)}."
        )
    args[CAPTION_ARG_INDEX] = caption
    args[LYRICS_ARG_INDEX] = lyrics
    if is_formatted:
        args[IS_FORMAT_CAPTION_ARG_INDEX] = True
    args[AUTOGEN_ARG_INDEX] = False
    args[CURRENT_BATCH_INDEX_ARG_INDEX] = 0
    args[TOTAL_BATCHES_ARG_INDEX] = 1
    args[BATCH_QUEUE_ARG_INDEX] = {}
    args[GENERATION_PARAMS_STATE_ARG_INDEX] = {}
    return args


def extract_generation_status(result: Any) -> str:
    """Return the status string from a generation result tuple when available."""

    if isinstance(result, tuple) and len(result) > 10:
        return str(result[10] or "").strip()
    return ""


def extract_generation_paths(result: Any) -> list[str]:
    """Return generated artifact paths from a final generation result tuple."""

    if not isinstance(result, tuple) or len(result) <= 8:
        return []
    paths = result[8]
    if not isinstance(paths, list):
        return []
    return [str(path) for path in paths if path]


def get_lm_format_settings(generation_args: Sequence[Any]) -> dict[str, Any]:
    """Return LM formatting settings from the current generation controls."""

    return {
        "bpm": generation_args[BPM_ARG_INDEX],
        "audio_duration": generation_args[AUDIO_DURATION_ARG_INDEX],
        "key_scale": generation_args[KEY_SCALE_ARG_INDEX],
        "time_signature": generation_args[TIME_SIGNATURE_ARG_INDEX],
        "lm_temperature": generation_args[LM_TEMPERATURE_ARG_INDEX],
        "lm_top_k": generation_args[LM_TOP_K_ARG_INDEX],
        "lm_top_p": generation_args[LM_TOP_P_ARG_INDEX],
        "lm_model_path": generation_args[LM_MODEL_PATH_ARG_INDEX],
        "backend": generation_args[BACKEND_ARG_INDEX],
        "device": generation_args[DEVICE_ARG_INDEX],
        "offload_to_cpu": bool(generation_args[OFFLOAD_TO_CPU_ARG_INDEX]),
    }
