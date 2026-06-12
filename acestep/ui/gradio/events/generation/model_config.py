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
    TASK_TYPES_SFT,
    TASK_TYPES_BASE,
    GENERATION_MODES_TURBO,
    GENERATION_MODES_SFT,
    GENERATION_MODES_BASE,
    MODE_TO_TASK_TYPE,
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

_UNSUPPORTED_MODE_REASONS = {
    "Extract": "Base only",
    "Lego": "Base only",
    "Complete": "Base only",
}

_RECOMMENDED_MODE_LABELS = {
    "Remix": "Remix (SFT Model Recommended)",
    "Repaint": "Repaint (SFT Model Recommended)",
    "Lego": "Lego (Base Model Recommended)",
    "Complete": "Complete (Base Model Recommended)",
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
        Ten-element tuple of ``gr.update()`` dicts for inference_steps,
        guidance_scale, use_adg, shift, cfg_interval_start, cfg_interval_end,
        task_type, generation_mode, init_llm_checkbox, and dcw_enabled.
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


def _choice_value(choice: str | tuple[str, str]) -> str:
    """Return the semantic mode value from a raw or display choice."""

    if isinstance(choice, tuple):
        return str(choice[1])
    return str(choice)


def get_supported_generation_modes(
    is_turbo: bool,
    is_pure_base: bool = False,
    is_sft: bool = False,
) -> list[str]:
    """Return generation modes supported by a model family."""

    if is_pure_base:
        return list(GENERATION_MODES_BASE)
    if is_sft and not is_turbo:
        return list(GENERATION_MODES_SFT)
    return list(GENERATION_MODES_TURBO)


def get_generation_mode_display_choices(
    supported_modes: list[str],
) -> list[str | tuple[str, str]]:
    """Return radio choices that show unavailable modes without enabling them."""

    supported = set(supported_modes)
    choices: list[str | tuple[str, str]] = []
    for mode in GENERATION_MODES_BASE:
        label = _RECOMMENDED_MODE_LABELS.get(mode, mode)
        if mode in supported:
            choices.append((label, mode) if label != mode else mode)
            continue
        reason = _UNSUPPORTED_MODE_REASONS.get(mode, "not supported")
        choices.append((f"{label} - unavailable ({reason})", mode))
    return choices


def get_generation_mode_choices_for_path(config_path: str | None) -> list[str | tuple[str, str]]:
    """Return model-aware generation-mode choices for the Gradio selector."""

    cfg = get_ui_control_config_for_path(config_path)
    return list(cfg["generation_mode_choices"])


def get_supported_generation_modes_for_path(config_path: str | None) -> list[str]:
    """Return raw supported mode values for a model config path."""

    cfg = get_ui_control_config_for_path(config_path)
    return list(cfg["supported_generation_modes"])


def is_generation_mode_supported_for_path(mode: str, config_path: str | None) -> bool:
    """Return whether a mode can be used with the selected model config."""

    return mode in get_supported_generation_modes_for_path(config_path)


def fallback_generation_mode_for_path(
    mode: str | None,
    config_path: str | None,
) -> str:
    """Return a supported mode, preserving *mode* when possible."""

    supported = get_supported_generation_modes_for_path(config_path)
    if mode in supported:
        return str(mode)
    return "Custom" if "Custom" in supported else supported[0]


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
    elif is_sft and not is_turbo:
        task_choices = TASK_TYPES_SFT
    else:
        task_choices = TASK_TYPES_TURBO
    is_pure_base_family = bool(is_pure_base)
    supported_modes = get_supported_generation_modes(
        is_turbo,
        is_pure_base=is_pure_base,
        is_sft=is_sft,
    )
    mode_choices = get_generation_mode_display_choices(supported_modes)

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
            "dcw_enabled_value": True,
            "cfg_interval_start_visible": False,
            "cfg_interval_end_value": 1.0,
            "cfg_interval_end_visible": False,
            "task_type_choices": task_choices,
            "generation_mode_choices": mode_choices,
            "supported_generation_modes": supported_modes,
            "is_pure_base_family": is_pure_base_family,
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
            "shift_value": 3.0,
            "shift_visible": True,
            "cfg_interval_start_value": 0.0,
            "dcw_enabled_value": False,
            "cfg_interval_start_visible": True,
            "cfg_interval_end_value": 1.0,
            "cfg_interval_end_visible": True,
            "task_type_choices": task_choices,
            "generation_mode_choices": mode_choices,
            "supported_generation_modes": supported_modes,
            "is_pure_base_family": is_pure_base_family,
        }

    # SFT and unknown non-turbo checkpoints default to the documented CFG
    # schedule with the same shift used by the official API defaults.
    return {
        **COMMON_DIFFUSION_CONTROL_RANGES,
        "inference_steps_value": 50,
        "inference_steps_maximum": 200,
        "inference_steps_minimum": 1,
        "guidance_scale_value": 7.0,
        "guidance_scale_visible": True,
        "use_adg_value": False,
        "use_adg_visible": True,
        "shift_value": 3.0,
        "shift_visible": True,
        "cfg_interval_start_value": 0.0,
        "dcw_enabled_value": False,
        "cfg_interval_start_visible": True,
        "cfg_interval_end_value": 1.0,
        "cfg_interval_end_visible": True,
        "task_type_choices": task_choices,
        "generation_mode_choices": mode_choices,
        "supported_generation_modes": supported_modes,
        "is_pure_base_family": is_pure_base_family,
    }


def get_model_type_ui_settings_from_config(cfg: dict, current_mode: str | None = None):
    """Get gr.update() tuple from an already-resolved UI configuration."""

    new_choices = cfg["generation_mode_choices"]
    supported_modes = cfg.get("supported_generation_modes", GENERATION_MODES_TURBO)
    new_choice_values = [_choice_value(choice) for choice in new_choices]
    resolved_mode = current_mode if current_mode in supported_modes else "Custom"
    if resolved_mode not in new_choice_values:
        resolved_mode = new_choice_values[0] if new_choice_values else "Custom"
    mode_update = gr.update(choices=new_choices, value=resolved_mode)
    task_type = MODE_TO_TASK_TYPE.get(resolved_mode, "text2music")
    init_llm_update = gr.update(value=False) if cfg.get("is_pure_base_family") else gr.update()
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
        task_type,
        mode_update,
        init_llm_update,
    )


def get_model_type_ui_settings(
    is_turbo: bool,
    current_mode: str | None = None,
    is_pure_base: bool = False,
    is_sft: bool = False,
):
    """Get gr.update() tuple for model-type controls.

    Args:
        is_turbo: Whether the model is a turbo variant.
        current_mode: Current generation mode value to preserve.
        is_pure_base: Whether the model is a pure base model.
        is_sft: Whether the model is an SFT variant.

    Returns:
        Tuple of updates for inference_steps, guidance_scale, use_adg,
        shift, cfg_interval_start, cfg_interval_end, task_type,
        generation_mode, and init_llm_checkbox.
    """
    cfg = get_ui_control_config(is_turbo, is_pure_base=is_pure_base, is_sft=is_sft)
    return get_model_type_ui_settings_from_config(cfg, current_mode=current_mode)


def get_generation_mode_choices(is_pure_base: bool = False, is_sft: bool = False) -> list:
    """Get the list of generation mode choices based on model type.

    Args:
        is_pure_base: Whether the model is a pure base model.
        is_sft: Whether the model is an SFT model.

    Returns:
        List of mode choice strings.
    """
    supported_modes = get_supported_generation_modes(
        False,
        is_pure_base=is_pure_base,
        is_sft=is_sft,
    )
    return get_generation_mode_display_choices(supported_modes)
