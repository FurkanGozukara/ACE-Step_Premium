"""LoRA training VRAM preset definitions for Gradio controls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LORA_VRAM_PRESET_MANUAL = "Manual"
LORA_VRAM_PRESET_8_TO_10GB = "8-10 GB"
LORA_VRAM_PRESET_10GB_PLUS = "10 GB+"
LORA_VRAM_PRESET_12_TO_16GB = "12-16 GB"
LORA_VRAM_PRESET_16_TO_24GB = "16-24 GB"
LORA_VRAM_PRESET_24GB_PLUS = "24GB+"

# Backward-compatible aliases for callers that import the older preset names.
LORA_VRAM_PRESET_10GB = LORA_VRAM_PRESET_8_TO_10GB
LORA_VRAM_PRESET_12GB = LORA_VRAM_PRESET_10GB_PLUS
LORA_VRAM_PRESET_16GB = LORA_VRAM_PRESET_12_TO_16GB
LORA_VRAM_PRESET_16GB_PLUS = LORA_VRAM_PRESET_16_TO_24GB
LORA_VRAM_PRESET_24GB = LORA_VRAM_PRESET_24GB_PLUS

DEFAULT_LORA_VRAM_PRESET = LORA_VRAM_PRESET_24GB
VRAM_10GB_PLUS_MIN_GB = 10.0
VRAM_16_TO_24GB_MIN_GB = 15.5
VRAM_24GB_PLUS_MIN_GB = 23.3

LORA_VRAM_PRESET_CHOICES = [
    LORA_VRAM_PRESET_MANUAL,
    LORA_VRAM_PRESET_10GB,
    LORA_VRAM_PRESET_12GB,
    LORA_VRAM_PRESET_16GB,
    LORA_VRAM_PRESET_16GB_PLUS,
    LORA_VRAM_PRESET_24GB,
]

_PRESETS: dict[str, dict[str, Any]] = {
    LORA_VRAM_PRESET_10GB: {
        "lora_rank": 16,
        "lora_alpha": 128,
        "gradient_checkpointing": True,
        "activation_cpu_offload": True,
        "offload_non_decoder": True,
        "keep_frozen_base_in_compute_dtype": True,
        "use_8bit_adam": True,
        "base_quantization": "FP8 scaled",
        "empty_cache_every_n_steps": 5,
    },
    LORA_VRAM_PRESET_12GB: {
        "lora_rank": 32,
        "lora_alpha": 128,
        "gradient_checkpointing": True,
        "activation_cpu_offload": True,
        "offload_non_decoder": True,
        "keep_frozen_base_in_compute_dtype": True,
        "use_8bit_adam": True,
        "base_quantization": "FP8 scaled",
        "empty_cache_every_n_steps": 5,
    },
    LORA_VRAM_PRESET_16GB: {
        "lora_rank": 64,
        "lora_alpha": 128,
        "gradient_checkpointing": True,
        "activation_cpu_offload": False,
        "offload_non_decoder": True,
        "keep_frozen_base_in_compute_dtype": True,
        "use_8bit_adam": True,
        "base_quantization": "Disabled",
        "empty_cache_every_n_steps": 10,
    },
    LORA_VRAM_PRESET_16GB_PLUS: {
        "lora_rank": 128,
        "lora_alpha": 128,
        "gradient_checkpointing": True,
        "activation_cpu_offload": False,
        "offload_non_decoder": True,
        "keep_frozen_base_in_compute_dtype": True,
        "use_8bit_adam": False,
        "base_quantization": "Disabled",
        "empty_cache_every_n_steps": 0,
    },
    LORA_VRAM_PRESET_24GB: {
        "lora_rank": 128,
        "lora_alpha": 128,
        "gradient_checkpointing": True,
        "activation_cpu_offload": False,
        "offload_non_decoder": True,
        "keep_frozen_base_in_compute_dtype": False,
        "use_8bit_adam": False,
        "base_quantization": "Disabled",
        "empty_cache_every_n_steps": 0,
    },
}


def get_lora_vram_preset(name: object) -> dict[str, Any]:
    """Return a copy of the named LoRA VRAM preset, or an empty dict for manual."""

    preset_name = str(name or "").strip()
    if preset_name == LORA_VRAM_PRESET_MANUAL:
        return {}
    return deepcopy(_PRESETS.get(preset_name, {}))


def select_lora_vram_preset_for_gpu(gpu_memory_gb: float) -> str:
    """Return the measured LoRA preset that best matches available GPU VRAM."""

    if gpu_memory_gb > VRAM_24GB_PLUS_MIN_GB:
        return LORA_VRAM_PRESET_24GB
    if gpu_memory_gb >= VRAM_16_TO_24GB_MIN_GB:
        return LORA_VRAM_PRESET_16GB_PLUS
    if gpu_memory_gb >= VRAM_10GB_PLUS_MIN_GB:
        return LORA_VRAM_PRESET_12GB
    return LORA_VRAM_PRESET_10GB


def default_lora_vram_preset_name() -> str:
    """Return the runtime default LoRA VRAM preset for the current GPU."""

    try:
        from acestep.gpu_config import get_global_gpu_config

        gpu_config = get_global_gpu_config()
        return select_lora_vram_preset_for_gpu(float(gpu_config.gpu_memory_gb))
    except Exception:
        return DEFAULT_LORA_VRAM_PRESET


def apply_lora_vram_preset(name: object, values: dict[str, Any]) -> dict[str, Any]:
    """Return submitted training values unchanged for compatibility callers."""

    _ = name
    return dict(values)
