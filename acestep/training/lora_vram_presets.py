"""Measured LoRA training VRAM preset definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LORA_VRAM_PRESET_MANUAL = "Manual"
LORA_VRAM_PRESET_10GB = "10 GB - experimental"
LORA_VRAM_PRESET_12GB = "12 GB - safest"
LORA_VRAM_PRESET_16GB = "16 GB - balanced"
LORA_VRAM_PRESET_24GB = "24 GB+ - faster"

DEFAULT_LORA_VRAM_PRESET = LORA_VRAM_PRESET_24GB
VRAM_12GB_CLASS_MIN_GB = 11.5
VRAM_16GB_CLASS_MIN_GB = 15.5
VRAM_24GB_CLASS_MIN_GB = 23.5

LORA_VRAM_PRESET_CHOICES = [
    LORA_VRAM_PRESET_MANUAL,
    LORA_VRAM_PRESET_10GB,
    LORA_VRAM_PRESET_12GB,
    LORA_VRAM_PRESET_16GB,
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
    LORA_VRAM_PRESET_24GB: {
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
}


def get_lora_vram_preset(name: object) -> dict[str, Any]:
    """Return a copy of the named LoRA VRAM preset, or an empty dict for manual."""

    preset_name = str(name or "").strip()
    if preset_name == LORA_VRAM_PRESET_MANUAL:
        return {}
    return deepcopy(_PRESETS.get(preset_name, {}))


def select_lora_vram_preset_for_gpu(gpu_memory_gb: float) -> str:
    """Return the measured LoRA preset that best matches available GPU VRAM."""

    if gpu_memory_gb >= VRAM_24GB_CLASS_MIN_GB:
        return LORA_VRAM_PRESET_24GB
    if gpu_memory_gb >= VRAM_16GB_CLASS_MIN_GB:
        return LORA_VRAM_PRESET_16GB
    if gpu_memory_gb >= VRAM_12GB_CLASS_MIN_GB:
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
    """Return training values with the selected preset overlaid."""

    preset = get_lora_vram_preset(name)
    if not preset:
        return dict(values)

    updated = dict(values)
    updated.update(preset)
    return updated
