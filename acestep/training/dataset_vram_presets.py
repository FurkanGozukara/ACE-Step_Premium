"""Measured dataset action VRAM preset definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DATASET_VRAM_PRESET_AUTO = "Auto (GPU default)"
DATASET_VRAM_PRESET_12GB = "10 GB+"
DATASET_VRAM_PRESET_16GB = "12-16 GB"
DATASET_VRAM_PRESET_24GB = "24 GB+ - quality"

DEFAULT_DATASET_VRAM_PRESET = DATASET_VRAM_PRESET_AUTO
VRAM_16GB_CLASS_MIN_GB = 15.5
VRAM_24GB_CLASS_MIN_GB = 23.5

DATASET_VRAM_PRESET_CHOICES = [
    DATASET_VRAM_PRESET_AUTO,
    DATASET_VRAM_PRESET_12GB,
    DATASET_VRAM_PRESET_16GB,
    DATASET_VRAM_PRESET_24GB,
]

_PRESETS: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    DATASET_VRAM_PRESET_12GB: {
        "auto_label": {
            "dit": {
                "offload_to_cpu": True,
                "offload_dit_to_cpu": True,
                "quantization": "fp8_scaled",
                "compile_model": False,
            },
            "llm": {
                "lm_model_path": "acestep-5Hz-lm-0.6B",
                "backend": "pt",
                "device": "auto",
                "offload_to_cpu": True,
            },
        },
        "preprocess": {
            "dit": {
                "offload_to_cpu": True,
                "offload_dit_to_cpu": False,
                "quantization": "int8_weight_only",
                "compile_model": False,
            },
        },
    },
    DATASET_VRAM_PRESET_16GB: {
        "auto_label": {
            "dit": {
                "offload_to_cpu": True,
                "offload_dit_to_cpu": False,
                "quantization": "int8_weight_only",
                "compile_model": False,
            },
            "llm": {
                "lm_model_path": "acestep-5Hz-lm-1.7B",
                "backend": "pt",
                "device": "auto",
                "offload_to_cpu": True,
            },
        },
        "preprocess": {
            "dit": {
                "offload_to_cpu": True,
                "offload_dit_to_cpu": False,
                "quantization": None,
                "compile_model": False,
            },
        },
    },
    DATASET_VRAM_PRESET_24GB: {
        "auto_label": {
            "dit": {
                "offload_to_cpu": True,
                "offload_dit_to_cpu": False,
                "quantization": None,
                "compile_model": False,
            },
            "llm": {
                "lm_model_path": "acestep-5Hz-lm-4B",
                "backend": "pt",
                "device": "auto",
                "offload_to_cpu": True,
            },
        },
        "preprocess": {
            "dit": {
                "offload_to_cpu": False,
                "offload_dit_to_cpu": False,
                "quantization": None,
                "compile_model": False,
            },
        },
    },
}


def get_dataset_vram_preset(name: object) -> dict[str, dict[str, dict[str, Any]]]:
    """Return a copy of the named dataset preset, or an empty dict for auto."""

    preset_name = str(name or "").strip()
    if preset_name == DATASET_VRAM_PRESET_AUTO:
        return {}
    return deepcopy(_PRESETS.get(preset_name, {}))


def select_dataset_vram_preset_for_gpu(gpu_memory_gb: float) -> str:
    """Return the measured dataset-action preset that best matches GPU VRAM."""

    if gpu_memory_gb >= VRAM_24GB_CLASS_MIN_GB:
        return DATASET_VRAM_PRESET_24GB
    if gpu_memory_gb >= VRAM_16GB_CLASS_MIN_GB:
        return DATASET_VRAM_PRESET_16GB
    return DATASET_VRAM_PRESET_12GB


def default_dataset_vram_preset_name() -> str:
    """Return the runtime default dataset-action VRAM preset for the current GPU."""

    try:
        from acestep.gpu_config import get_global_gpu_config

        gpu_config = get_global_gpu_config()
        return select_dataset_vram_preset_for_gpu(float(gpu_config.gpu_memory_gb))
    except Exception:
        return DEFAULT_DATASET_VRAM_PRESET


def dataset_vram_preset_requires_subprocess(name: object) -> bool:
    """Return whether the preset needs an isolated worker to control init settings."""

    return bool(get_dataset_vram_preset(name))


def apply_dataset_dit_preset(
    name: object,
    params: dict[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    """Return DiT init params with the selected dataset preset overlaid."""

    preset = get_dataset_vram_preset(name).get(operation, {}).get("dit", {})
    updated = dict(params)
    updated.update(preset)
    return updated


def apply_dataset_llm_preset(name: object, params: dict[str, Any]) -> dict[str, Any]:
    """Return LM init params with the selected auto-label preset overlaid."""

    preset = get_dataset_vram_preset(name).get("auto_label", {}).get("llm", {})
    updated = dict(params)
    updated.update(preset)
    return updated
