"""Lazy facade for generation event handlers.

This keeps Gradio startup free of heavyweight generation/runtime imports until
the corresponding event callback actually runs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from acestep.ui.gradio.events.dcw_defaults import (
    NON_THINK_DCW_DEFAULTS,
    THINK_DCW_DEFAULTS,
    get_dcw_defaults_for_think,
)


def _load_attr(module_name: str, attr_name: str) -> Any:
    """Import and return the requested handler attribute."""

    module = import_module(f"acestep.ui.gradio.events.generation.{module_name}")
    return getattr(module, attr_name)


def _forward(module_name: str, attr_name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve a handler lazily and invoke it."""

    return _load_attr(module_name, attr_name)(*args, **kwargs)


def clamp_duration_to_gpu_limit(*args: Any, **kwargs: Any) -> Any:
    return _forward("validation", "clamp_duration_to_gpu_limit", *args, **kwargs)


def parse_and_validate_timesteps(*args: Any, **kwargs: Any) -> Any:
    return _forward("validation", "parse_and_validate_timesteps", *args, **kwargs)


def _has_reference_audio(*args: Any, **kwargs: Any) -> Any:
    return _forward("validation", "_has_reference_audio", *args, **kwargs)


def _extract_audio_path(*args: Any, **kwargs: Any) -> Any:
    return _forward("validation", "_extract_audio_path", *args, **kwargs)


def validate_uploaded_audio_file(*args: Any, **kwargs: Any) -> Any:
    return _forward("validation", "validate_uploaded_audio_file", *args, **kwargs)


def _contains_audio_code_tokens(*args: Any, **kwargs: Any) -> Any:
    return _forward("validation", "_contains_audio_code_tokens", *args, **kwargs)


def load_metadata(*args: Any, **kwargs: Any) -> Any:
    return _forward("metadata_loading", "load_metadata", *args, **kwargs)


def load_random_example(*args: Any, **kwargs: Any) -> Any:
    return _forward("metadata_loading", "load_random_example", *args, **kwargs)


def sample_example_smart(*args: Any, **kwargs: Any) -> Any:
    return _forward("metadata_loading", "sample_example_smart", *args, **kwargs)


def load_random_simple_description(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "metadata_loading",
        "load_random_simple_description",
        *args,
        **kwargs,
    )


def refresh_checkpoints(*args: Any, **kwargs: Any) -> Any:
    return _forward("service_init", "refresh_checkpoints", *args, **kwargs)


def initial_lora_dropdown_choices(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_browser", "initial_lora_dropdown_choices", *args, **kwargs)


def refresh_lora_dropdown(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_browser", "refresh_lora_dropdown", *args, **kwargs)


def select_lora_dropdown_path(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_browser", "select_lora_dropdown_path", *args, **kwargs)


def update_lora_next_run_status(*args: Any, **kwargs: Any) -> Any:
    return _forward("lora_browser", "update_lora_next_run_status", *args, **kwargs)


def init_service_wrapper(*args: Any, **kwargs: Any) -> Any:
    return _forward("service_init", "init_service_wrapper", *args, **kwargs)


def on_tier_change(*args: Any, **kwargs: Any) -> Any:
    return _forward("service_init", "on_tier_change", *args, **kwargs)


def is_pure_base_model(*args: Any, **kwargs: Any) -> Any:
    return _forward("model_config", "is_pure_base_model", *args, **kwargs)


def is_sft_model(*args: Any, **kwargs: Any) -> Any:
    return _forward("model_config", "is_sft_model", *args, **kwargs)


def update_model_type_settings(*args: Any, **kwargs: Any) -> Any:
    return _forward("model_config", "update_model_type_settings", *args, **kwargs)


def get_ui_control_config(*args: Any, **kwargs: Any) -> Any:
    return _forward("model_config", "get_ui_control_config", *args, **kwargs)


def get_model_type_ui_settings(*args: Any, **kwargs: Any) -> Any:
    return _forward("model_config", "get_model_type_ui_settings", *args, **kwargs)


def get_generation_mode_choices(*args: Any, **kwargs: Any) -> Any:
    return _forward("model_config", "get_generation_mode_choices", *args, **kwargs)


def compute_mode_ui_updates(*args: Any, **kwargs: Any) -> Any:
    return _forward("mode_ui", "compute_mode_ui_updates", *args, **kwargs)


def handle_generation_mode_change(*args: Any, **kwargs: Any) -> Any:
    return _forward("mode_ui", "handle_generation_mode_change", *args, **kwargs)


def handle_extract_track_name_change(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "mode_ui",
        "handle_extract_track_name_change",
        *args,
        **kwargs,
    )


def handle_extract_src_audio_change(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "mode_ui",
        "handle_extract_src_audio_change",
        *args,
        **kwargs,
    )


def handle_create_sample(*args: Any, **kwargs: Any) -> Any:
    return _forward("llm_actions", "handle_create_sample", *args, **kwargs)


def handle_format_sample(*args: Any, **kwargs: Any) -> Any:
    return _forward("llm_actions", "handle_format_sample", *args, **kwargs)


def handle_format_caption(*args: Any, **kwargs: Any) -> Any:
    return _forward("llm_actions", "handle_format_caption", *args, **kwargs)


def handle_format_lyrics(*args: Any, **kwargs: Any) -> Any:
    return _forward("llm_actions", "handle_format_lyrics", *args, **kwargs)


def transcribe_audio_codes(*args: Any, **kwargs: Any) -> Any:
    return _forward("llm_actions", "transcribe_audio_codes", *args, **kwargs)


def analyze_src_audio(*args: Any, **kwargs: Any) -> Any:
    return _forward("llm_actions", "analyze_src_audio", *args, **kwargs)


def update_dcw_defaults_for_think(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "update_dcw_defaults_for_think", *args, **kwargs)


def update_negative_prompt_visibility(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "update_negative_prompt_visibility",
        *args,
        **kwargs,
    )


def on_auto_checkbox_change(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "on_auto_checkbox_change", *args, **kwargs)


def reset_all_auto(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "reset_all_auto", *args, **kwargs)


def uncheck_auto_for_populated_fields(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "uncheck_auto_for_populated_fields",
        *args,
        **kwargs,
    )


def update_audio_cover_strength_visibility(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "update_audio_cover_strength_visibility",
        *args,
        **kwargs,
    )


def convert_src_audio_to_codes_wrapper(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "convert_src_audio_to_codes_wrapper",
        *args,
        **kwargs,
    )


def update_instruction_ui(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "update_instruction_ui", *args, **kwargs)


def update_transcribe_button_text(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "update_transcribe_button_text", *args, **kwargs)


def reset_format_caption_flag(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "reset_format_caption_flag", *args, **kwargs)


def update_audio_uploads_accordion(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "update_audio_uploads_accordion",
        *args,
        **kwargs,
    )


def handle_instrumental_checkbox(*args: Any, **kwargs: Any) -> Any:
    return _forward("ui_helpers", "handle_instrumental_checkbox", *args, **kwargs)


def handle_simple_instrumental_change(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "handle_simple_instrumental_change",
        *args,
        **kwargs,
    )


def update_audio_components_visibility(*args: Any, **kwargs: Any) -> Any:
    return _forward(
        "ui_helpers",
        "update_audio_components_visibility",
        *args,
        **kwargs,
    )


__all__ = [
    "clamp_duration_to_gpu_limit",
    "parse_and_validate_timesteps",
    "_has_reference_audio",
    "_extract_audio_path",
    "validate_uploaded_audio_file",
    "_contains_audio_code_tokens",
    "load_metadata",
    "load_random_example",
    "sample_example_smart",
    "load_random_simple_description",
    "refresh_checkpoints",
    "initial_lora_dropdown_choices",
    "refresh_lora_dropdown",
    "select_lora_dropdown_path",
    "update_lora_next_run_status",
    "init_service_wrapper",
    "on_tier_change",
    "is_pure_base_model",
    "is_sft_model",
    "update_model_type_settings",
    "get_ui_control_config",
    "get_model_type_ui_settings",
    "get_generation_mode_choices",
    "compute_mode_ui_updates",
    "handle_generation_mode_change",
    "handle_extract_track_name_change",
    "handle_extract_src_audio_change",
    "handle_create_sample",
    "handle_format_sample",
    "handle_format_caption",
    "handle_format_lyrics",
    "transcribe_audio_codes",
    "analyze_src_audio",
    "NON_THINK_DCW_DEFAULTS",
    "THINK_DCW_DEFAULTS",
    "get_dcw_defaults_for_think",
    "update_dcw_defaults_for_think",
    "update_negative_prompt_visibility",
    "on_auto_checkbox_change",
    "reset_all_auto",
    "uncheck_auto_for_populated_fields",
    "update_audio_cover_strength_visibility",
    "convert_src_audio_to_codes_wrapper",
    "update_instruction_ui",
    "update_transcribe_button_text",
    "reset_format_caption_flag",
    "update_audio_uploads_accordion",
    "handle_instrumental_checkbox",
    "handle_simple_instrumental_change",
    "update_audio_components_visibility",
]
