"""Ordered Load Metadata value construction for generation fields."""

from __future__ import annotations

from typing import Any

import gradio as gr

from .audio_format_options import (
    DEFAULT_MP3_BITRATE,
    MP3_BITRATE_CHOICES,
    MP3_SAMPLE_RATE_CHOICES,
    mp3_controls_visible,
)
from .generation_count import normalize_generation_count
from .metadata_fields import LOAD_METADATA_GENERATION_OUTPUT_KEYS
from .metadata_restore_aliases import (
    apply_audio_format_values,
    apply_extract_values,
    apply_runtime_values,
    apply_simple_aliases,
)
from .metadata_restore_defaults import default_metadata_values
from .metadata_restore_document import (
    coerce_duration,
    coerce_optional_int,
    first_value,
    mapping,
    mode_for_task_type,
    runtime_payload,
)
from .metadata_restore_settings import audio_processing_ui_values, sam_audio_ui_values

_FILE_OUTPUT_KEYS = (
    "reference_audio",
    "src_audio",
    "ap_diffpitcher_reference_audio",
    "ap_diffpitcher_midi",
)


def unchanged_metadata_values() -> list[Any]:
    """Return no-op Gradio updates matching the metadata output contract."""

    return [gr.update() for _ in LOAD_METADATA_GENERATION_OUTPUT_KEYS]


def metadata_values_from_payload(
    payload: dict[str, Any],
    llm_handler: Any = None,
) -> list[Any]:
    """Return ordered component values for a normalized metadata payload."""

    values = default_metadata_values()
    generation_params = mapping(payload.get("generation_params"))
    generation_config = mapping(payload.get("generation_config"))
    runtime = runtime_payload(payload)
    saved_assets = mapping(payload.get("saved_run_assets"))

    task_type = first_value(payload, generation_params, "task_type", default="text2music")
    values["task_type"] = task_type
    values["generation_mode"] = mode_for_task_type(task_type)
    values["captions"] = first_value(
        payload,
        generation_params,
        "captions",
        "caption",
        default="",
    )
    values["lyrics"] = first_value(payload, generation_params, "lyrics", default="")
    values["vocal_language"] = first_value(
        payload,
        generation_params,
        "vocal_language",
        "language",
        default="unknown",
    )
    values["bpm"] = coerce_optional_int(first_value(payload, generation_params, "bpm"))
    values["key_scale"] = first_value(
        payload,
        generation_params,
        "key_scale",
        "keyscale",
        default="",
    )
    values["time_signature"] = first_value(
        payload,
        generation_params,
        "time_signature",
        "timesignature",
        default="",
    )
    values["audio_duration"] = coerce_duration(
        first_value(payload, generation_params, "audio_duration", "duration", default=-1),
        llm_handler,
    )
    values["batch_size_input"] = normalize_generation_count(
        first_value(
            payload,
            generation_config,
            "requested_generation_count",
            "generation_count",
            "batch_size",
            default=1,
        )
    )
    values["inference_steps"] = first_value(
        payload,
        generation_params,
        "inference_steps_requested",
        "inference_steps",
        default=8,
    )
    values["guidance_scale"] = first_value(
        payload,
        generation_params,
        "guidance_scale",
        default=7.0,
    )
    values["seed"] = first_value(payload, "seed_input", "seed", default="-1")
    values["random_seed_checkbox"] = bool(
        first_value(payload, generation_config, "random_seed_checkbox", default=False)
    )
    values["reference_audio"] = first_value(
        saved_assets,
        payload,
        "reference_audio_path",
        "reference_audio",
        default=None,
    )
    values["src_audio"] = first_value(
        saved_assets,
        payload,
        "source_audio_path",
        "src_audio",
        default=None,
    )
    apply_simple_aliases(values, payload, generation_params)
    apply_audio_format_values(values, payload, task_type)
    apply_extract_values(values, payload, generation_params, task_type)
    apply_runtime_values(values, runtime)
    values.update(audio_processing_ui_values(payload.get("audio_processing_settings")))
    values.update(sam_audio_ui_values(payload.get("sam_audio_settings")))
    _clear_file_output_values(values)
    _finalize_audio_code_state(values)
    _finalize_mp3_controls(values)
    return [values[key] for key in LOAD_METADATA_GENERATION_OUTPUT_KEYS]


def _clear_file_output_values(values: dict[str, Any]) -> None:
    """Do not restore Gradio file uploads from saved metadata."""

    for key in _FILE_OUTPUT_KEYS:
        values[key] = None


def _finalize_audio_code_state(values: dict[str, Any]) -> None:
    """Disable model thinking when restored metadata includes explicit audio codes."""

    if values["think_checkbox"] and str(values["text2music_audio_code_string"] or "").strip():
        values["think_checkbox"] = False


def _finalize_mp3_controls(values: dict[str, Any]) -> None:
    """Set Gradio update values for MP3 controls from restored audio format."""

    is_mp3 = mp3_controls_visible(values["audio_format"])
    values["mp3_controls_row"] = gr.update(visible=is_mp3)
    values["mp3_bitrate"] = gr.update(
        choices=MP3_BITRATE_CHOICES,
        value=values.get("mp3_bitrate") or DEFAULT_MP3_BITRATE,
        visible=is_mp3,
    )
    values["mp3_sample_rate"] = gr.update(
        choices=MP3_SAMPLE_RATE_CHOICES,
        value=values["mp3_sample_rate"],
        visible=is_mp3,
    )
