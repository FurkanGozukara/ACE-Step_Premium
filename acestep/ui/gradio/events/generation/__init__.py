"""Generation event-handler package.

This package intentionally avoids eager re-exports so Gradio startup can load
only the submodules that are actually needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ATTR_MODULES = {
    "clamp_duration_to_gpu_limit": "validation",
    "parse_and_validate_timesteps": "validation",
    "_has_reference_audio": "validation",
    "_extract_audio_path": "validation",
    "validate_uploaded_audio_file": "validation",
    "_contains_audio_code_tokens": "validation",
    "load_metadata": "metadata_loading",
    "load_random_example": "metadata_loading",
    "sample_example_smart": "metadata_loading",
    "load_random_simple_description": "metadata_loading",
    "refresh_checkpoints": "service_init",
    "init_service_wrapper": "service_init",
    "on_tier_change": "service_init",
    "is_pure_base_model": "model_config",
    "is_sft_model": "model_config",
    "update_model_type_settings": "model_config",
    "get_ui_control_config": "model_config",
    "get_model_type_ui_settings": "model_config",
    "get_generation_mode_choices": "model_config",
    "compute_mode_ui_updates": "mode_ui",
    "handle_generation_mode_change": "mode_ui",
    "handle_extract_track_name_change": "mode_ui",
    "handle_extract_src_audio_change": "mode_ui",
    "handle_create_sample": "llm_actions",
    "handle_format_sample": "llm_actions",
    "handle_format_caption": "llm_actions",
    "handle_format_lyrics": "llm_actions",
    "transcribe_audio_codes": "llm_actions",
    "analyze_src_audio": "llm_actions",
    "NON_THINK_DCW_DEFAULTS": "ui_helpers",
    "THINK_DCW_DEFAULTS": "ui_helpers",
    "get_dcw_defaults_for_think": "ui_helpers",
    "update_dcw_defaults_for_think": "ui_helpers",
    "update_negative_prompt_visibility": "ui_helpers",
    "on_auto_checkbox_change": "ui_helpers",
    "reset_all_auto": "ui_helpers",
    "uncheck_auto_for_populated_fields": "ui_helpers",
    "update_audio_cover_strength_visibility": "ui_helpers",
    "convert_src_audio_to_codes_wrapper": "ui_helpers",
    "update_instruction_ui": "ui_helpers",
    "update_transcribe_button_text": "ui_helpers",
    "reset_format_caption_flag": "ui_helpers",
    "update_audio_uploads_accordion": "ui_helpers",
    "handle_instrumental_checkbox": "ui_helpers",
    "handle_simple_instrumental_change": "ui_helpers",
    "update_audio_components_visibility": "ui_helpers",
}

__all__ = list(_ATTR_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve compatibility re-exports lazily."""

    module_name = _ATTR_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
