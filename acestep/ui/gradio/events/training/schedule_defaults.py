"""Model-aware schedule defaults for Gradio training controls."""

from __future__ import annotations

import re

import gradio as gr

from acestep.ui.gradio.events.generation.model_config import (
    get_ui_control_config_for_path,
)

_TRAINING_SCHEDULE_DEFAULTS = {
    "turbo": {"shift": 3.0, "num_inference_steps": 8},
    "base": {"shift": 1.0, "num_inference_steps": 50},
    "sft": {"shift": 1.0, "num_inference_steps": 50},
}


def training_schedule_defaults_for_model(model_config: str | None) -> dict[str, float | int]:
    """Return default training schedule values for a selected base model.

    Args:
        model_config: Selected DiT model path or identifier.

    Returns:
        A mapping containing training schedule values and visible slider ranges.
    """

    selected_model = model_config or "turbo"
    cfg = get_ui_control_config_for_path(selected_model)
    schedule = _training_schedule_for_model(selected_model)
    return {
        "shift": float(schedule["shift"]),
        "shift_minimum": float(cfg["shift_minimum"]),
        "shift_maximum": float(cfg["shift_maximum"]),
        "shift_step": float(cfg["shift_step"]),
        "num_inference_steps": int(schedule["num_inference_steps"]),
        "num_inference_steps_minimum": 1,
        "num_inference_steps_maximum": int(cfg["inference_steps_server_maximum"]),
    }


def training_schedule_updates_for_model(model_config: str | None) -> tuple[dict, dict]:
    """Return Gradio updates for model-specific training schedule controls."""

    defaults = training_schedule_defaults_for_model(model_config)
    return (
        gr.update(
            value=defaults["shift"],
            minimum=defaults["shift_minimum"],
            maximum=defaults["shift_maximum"],
            step=defaults["shift_step"],
        ),
        gr.update(
            value=defaults["num_inference_steps"],
            minimum=defaults["num_inference_steps_minimum"],
            maximum=defaults["num_inference_steps_maximum"],
            step=1,
        ),
    )


def _training_schedule_for_model(model_config: str) -> dict[str, float | int]:
    """Return training metadata defaults for a model path or identifier."""

    model_config_lower = model_config.lower()
    if _has_model_token("turbo", model_config_lower):
        return _TRAINING_SCHEDULE_DEFAULTS["turbo"]
    if _has_model_token("base", model_config_lower) and not _has_model_token(
        "sft", model_config_lower
    ):
        return _TRAINING_SCHEDULE_DEFAULTS["base"]
    if _has_model_token("sft", model_config_lower):
        return _TRAINING_SCHEDULE_DEFAULTS["sft"]
    return _TRAINING_SCHEDULE_DEFAULTS["sft"]


def _has_model_token(token: str, model_config_lower: str) -> bool:
    """Return whether a model token appears in a path-safe delimited form."""

    return (
        re.search(rf"(^|[\\\\/._-]){token}($|[\\\\/._-])", model_config_lower)
        is not None
    )
