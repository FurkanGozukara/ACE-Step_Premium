"""Quantization selection helpers shared by Gradio generation handlers."""

from __future__ import annotations

from typing import Any

from loguru import logger


QUANTIZATION_CHOICES = [
    ("Disabled", "none"),
    ("INT4 weight-only (experimental)", "int4_weight_only"),
    ("INT8 weight-only", "int8_weight_only"),
    ("FP8 weight-only", "fp8_weight_only"),
    ("FP8 scaled cache", "fp8_scaled"),
    ("W8A8 dynamic", "w8a8_dynamic"),
]


def default_quantization_value(value: Any) -> str:
    """Normalize a persisted/default quantization value for a dropdown."""

    if value in (None, False, "", "none", "null"):
        return "none"
    if value is True:
        return "int8_weight_only"
    normalized = str(value).strip().lower()
    allowed = {choice_value for _, choice_value in QUANTIZATION_CHOICES}
    return normalized if normalized in allowed else "none"


def select_quantization_value(selection: Any, *, device: str) -> str | None:
    """Return the canonical backend quantization value for a UI selection."""

    value = default_quantization_value(selection)
    if value == "none":
        return None
    if value != "int8_weight_only":
        return value
    if device not in {"auto", "cuda"}:
        return value

    try:
        import torch
    except ImportError:
        return value

    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                logger.info(
                    "Pre-Ampere CUDA detected: using w8a8_dynamic quantization for stability"
                )
                return "w8a8_dynamic"
    except Exception:
        return value
    return value
