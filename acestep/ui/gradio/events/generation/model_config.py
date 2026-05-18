"""Model configuration and UI control settings for generation handlers.

Contains functions for determining model type (turbo/base/pure-base),
producing UI control configurations, and computing gr.update() tuples
for model-type-dependent controls.
"""

import re
from collections.abc import Iterable

import gradio as gr

from acestep.constants import (
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
    GENERATION_MODES_TURBO,
    GENERATION_MODES_BASE,
)
from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
    SOURCE_BASE_DIT_MODEL,
    SOURCE_PREMIUM_DIT_MODEL,
    SOURCE_TURBO_DIT_MODEL,
)


PREFERRED_DIT_MODEL_ORDER = (
    DEFAULT_TURBO_DIT_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_BASE_DIT_MODEL,
    SOURCE_TURBO_DIT_MODEL,
    SOURCE_PREMIUM_DIT_MODEL,
    SOURCE_BASE_DIT_MODEL,
    "acestep-v15-sft",
    "acestep-v15-base",
    "acestep-v15-turbo",
)

COMMON_DIFFUSION_CONTROL_RANGES = {
    "inference_steps_server_maximum": 200,
    "guidance_scale_minimum": 1.0,
    "guidance_scale_maximum": 15.0,
    "guidance_scale_step": 0.1,
    "shift_minimum": 1.0,
    "shift_maximum": 5.0,
    "shift_step": 0.1,
    "cfg_interval_start_minimum": 0.0,
    "cfg_interval_start_maximum": 1.0,
    "cfg_interval_start_step": 0.01,
    "cfg_interval_end_minimum": 0.0,
    "cfg_interval_end_maximum": 1.0,
    "cfg_interval_end_step": 0.01,
}


def _has_token(token: str, path: str) -> bool:
    """Check if *token* appears as a delimited word in *path*.

    Matches when *token* is bounded by start/end of string or a common
    path delimiter (``/``, ``\\``, ``.``, ``_``, ``-``).
    """
    return re.search(rf"(^|[\\\\/._-]){token}($|[\\\\/._-])", path) is not None


def is_pure_base_model(config_path_lower: str) -> bool:
    """Check whether a model path refers to a pure base model.

    Args:
        config_path_lower: Lowercased model config path string.

    Returns:
        ``True`` when the path contains ``"base"`` and excludes ``"sft"`` and ``"turbo"``.
    """
    return (
        _has_token("base", config_path_lower)
        and not _has_token("sft", config_path_lower)
        and not _has_token("turbo", config_path_lower)
    )


def update_model_type_settings(config_path: str | None, current_mode: str | None = None) -> tuple:
    """Update UI settings based on model type (fallback when handler not initialized yet).

    Args:
        config_path: Model config path string.
        current_mode: Current generation mode value to preserve across choices update.

    Returns:
        Nine-element tuple of ``gr.update()`` dicts for inference_steps,
        guidance_scale, use_adg, shift, cfg_interval_start, cfg_interval_end,
        task_type, generation_mode, and init_llm_checkbox.
    """
    cfg = get_ui_control_config_for_path(config_path)
    return get_model_type_ui_settings_from_config(cfg, current_mode=current_mode)


def is_sft_model(config_path_lower: str) -> bool:
    """Check whether a model path refers to an SFT (supervised fine-tuned) model.

    Args:
        config_path_lower: Lowercased model config path string.

    Returns:
        ``True`` when the path contains ``"sft"`` and excludes ``"turbo"``.
    """
    return _has_token("sft", config_path_lower) and not _has_token("turbo", config_path_lower)


def is_xl_model(config_path_lower: str) -> bool:
    """Check whether a model path refers to an XL (4B DiT) variant.

    Args:
        config_path_lower: Lowercased model config path string.

    Returns:
        ``True`` when the path contains ``"xl"`` as a delimited token.
    """
    return _has_token("xl", config_path_lower)


def select_preferred_model_path(available_models: Iterable[str] | None) -> str:
    """Return the best default DiT model from the discovered model list."""

    models = [str(model) for model in (available_models or []) if str(model)]
    for preferred in PREFERRED_DIT_MODEL_ORDER:
        if preferred in models:
            return preferred
    return models[0] if models else DEFAULT_TURBO_DIT_MODEL


def get_ui_control_config_for_path(config_path: str | None) -> dict:
    """Return UI control configuration for a model config path string."""

    config_path_lower = (config_path or "").lower()
    return get_ui_control_config(
        _has_token("turbo", config_path_lower),
        is_pure_base=is_pure_base_model(config_path_lower),
        is_sft=is_sft_model(config_path_lower),
    )


def get_ui_control_config(is_turbo: bool, is_pure_base: bool = False, is_sft: bool = False) -> dict:
    """Return UI control configuration (values, limits, visibility) for model type.

    Args:
        is_turbo: Whether the model is a turbo variant.
        is_pure_base: Whether the model is a pure base model.
        is_sft: Whether the model is an SFT (supervised fine-tuned) variant.
              SFT uses the documented 50-step CFG schedule. Base uses the
              64-step APG/CFG schedule while keeping optional ADG user-selectable.

    Used by both interactive init and service-mode startup so controls stay consistent.
    """
    # Precedence: turbo > SFT > pure base > fallback.
    if is_pure_base:
        task_choices = TASK_TYPES_BASE
        mode_choices = GENERATION_MODES_BASE
    else:
        task_choices = TASK_TYPES_TURBO
        mode_choices = GENERATION_MODES_TURBO

    if is_turbo:
        return {
            **COMMON_DIFFUSION_CONTROL_RANGES,
            "inference_steps_value": 8,
            "inference_steps_maximum": 20,
            "inference_steps_minimum": 1,
            "guidance_scale_value": 1.0,
            "guidance_scale_visible": False,
            "use_adg_value": False,
            "use_adg_visible": False,
            "shift_value": 3.0,
            "shift_visible": True,
            "cfg_interval_start_value": 0.0,
            "cfg_interval_start_visible": False,
            "cfg_interval_end_value": 1.0,
            "cfg_interval_end_visible": False,
            "task_type_choices": task_choices,
            "generation_mode_choices": mode_choices,
        }
    if is_pure_base:
        # Keep Base on the same APG/CFG path used by the official pipeline.
        # The optional ADG branch remains user-selectable, but is not default.
        return {
            **COMMON_DIFFUSION_CONTROL_RANGES,
            "inference_steps_value": 64,
            "inference_steps_maximum": 200,
            "inference_steps_minimum": 1,
            "guidance_scale_value": 7.0,
            "guidance_scale_visible": True,
            "use_adg_value": False,
            "use_adg_visible": True,
            "shift_value": 1.0,
            "shift_visible": True,
            "cfg_interval_start_value": 0.0,
            "cfg_interval_start_visible": True,
            "cfg_interval_end_value": 1.0,
            "cfg_interval_end_visible": True,
            "task_type_choices": task_choices,
            "generation_mode_choices": mode_choices,
        }

    # SFT and unknown non-turbo checkpoints default to the documented CFG
    # schedule. Base/SFT use the model's native continuous shift=1.0 schedule.
    return {
        **COMMON_DIFFUSION_CONTROL_RANGES,
        "inference_steps_value": 50,
        "inference_steps_maximum": 200,
        "inference_steps_minimum": 1,
        "guidance_scale_value": 7.0,
        "guidance_scale_visible": True,
        "use_adg_value": False,
        "use_adg_visible": True,
        "shift_value": 1.0,
        "shift_visible": True,
        "cfg_interval_start_value": 0.0,
        "cfg_interval_start_visible": True,
        "cfg_interval_end_value": 1.0,
        "cfg_interval_end_visible": True,
        "task_type_choices": task_choices,
        "generation_mode_choices": mode_choices,
    }


def get_model_type_ui_settings_from_config(cfg: dict, current_mode: str | None = None):
    """Get gr.update() tuple from an already-resolved UI configuration."""

    new_choices = cfg["generation_mode_choices"]
    if current_mode and current_mode in new_choices:
        mode_update = gr.update(choices=new_choices, value=current_mode)
    else:
        mode_update = gr.update(choices=new_choices)
    init_llm_update = gr.update(value=False) if new_choices == GENERATION_MODES_BASE else gr.update()
    return (
        gr.update(
            value=cfg["inference_steps_value"],
            maximum=cfg["inference_steps_maximum"],
            minimum=cfg["inference_steps_minimum"],
        ),
        gr.update(
            value=cfg["guidance_scale_value"],
            minimum=cfg["guidance_scale_minimum"],
            maximum=cfg["guidance_scale_maximum"],
            step=cfg["guidance_scale_step"],
            visible=cfg["guidance_scale_visible"],
        ),
        gr.update(value=cfg["use_adg_value"], visible=cfg["use_adg_visible"]),
        gr.update(
            value=cfg["shift_value"],
            minimum=cfg["shift_minimum"],
            maximum=cfg["shift_maximum"],
            step=cfg["shift_step"],
            visible=cfg["shift_visible"],
        ),
        gr.update(
            value=cfg["cfg_interval_start_value"],
            minimum=cfg["cfg_interval_start_minimum"],
            maximum=cfg["cfg_interval_start_maximum"],
            step=cfg["cfg_interval_start_step"],
            visible=cfg["cfg_interval_start_visible"],
        ),
        gr.update(
            value=cfg["cfg_interval_end_value"],
            minimum=cfg["cfg_interval_end_minimum"],
            maximum=cfg["cfg_interval_end_maximum"],
            step=cfg["cfg_interval_end_step"],
            visible=cfg["cfg_interval_end_visible"],
        ),
        gr.update(),  # task_type
        mode_update,
        init_llm_update,
    )


def get_model_type_ui_settings(is_turbo: bool, current_mode: str | None = None, is_pure_base: bool = False, is_sft: bool = False):
    """Get gr.update() tuple for model-type controls.

    Args:
        is_turbo: Whether the model is a turbo variant.
        current_mode: Current generation mode value to preserve.
        is_pure_base: Whether the model is a pure base model.
        is_sft: Whether the model is an SFT variant.

    Returns:
        Tuple of updates for inference_steps, guidance_scale, use_adg,
        shift, cfg_interval_start, cfg_interval_end, task_type,
        generation_mode, init_llm_checkbox.
    """
    cfg = get_ui_control_config(is_turbo, is_pure_base=is_pure_base, is_sft=is_sft)
    return get_model_type_ui_settings_from_config(cfg, current_mode=current_mode)


def get_generation_mode_choices(is_pure_base: bool = False) -> list:
    """Get the list of generation mode choices based on model type.

    Args:
        is_pure_base: Whether the model is a pure base model.

    Returns:
        List of mode choice strings.
    """
    if is_pure_base:
        return GENERATION_MODES_BASE
    else:
        return GENERATION_MODES_TURBO
