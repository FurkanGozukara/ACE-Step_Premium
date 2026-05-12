"""Shared Gradio updates for model-dependent generation defaults."""

from typing import Any

import gradio as gr

from ...premium_features import model_quality_defaults


ADVANCED_MODEL_RESET_KEYS: tuple[str, ...] = (
    "infer_method",
    "sampler_mode",
    "velocity_norm_threshold",
    "velocity_ema_factor",
    "custom_timesteps",
    "dcw_wavelet",
)


def build_advanced_model_reset_updates(model_path: str | None) -> tuple[Any, ...]:
    """Return updates for advanced-only controls reset by model selection.

    Args:
        model_path: Selected DiT model path or model identifier.

    Returns:
        Update payloads ordered to match ``ADVANCED_MODEL_RESET_KEYS``.
    """

    defaults = model_quality_defaults(model_path)
    return tuple(
        gr.update(value=defaults[key])
        for key in ADVANCED_MODEL_RESET_KEYS
    )
