"""VRAM preset definitions for SAM-Audio inference."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SAM_VRAM_PRESET_32GB = "32gb_quality"
SAM_VRAM_PRESET_24GB = "24gb_balanced"
SAM_VRAM_PRESET_16GB = "16gb_fp8"
SAM_VRAM_PRESET_12GB = "12gb_lite_fp8"
SAM_VRAM_PRESET_10GB = "10gb_lite_fp8"
SAM_VRAM_PRESET_8GB = "8gb_cpu_lite"

SAM_VRAM_PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    ("32GB+ Quality Chunked", SAM_VRAM_PRESET_32GB),
    ("24GB Balanced Chunked", SAM_VRAM_PRESET_24GB),
    ("16GB FP8 Chunked", SAM_VRAM_PRESET_16GB),
    ("12GB Lite FP8 Chunked", SAM_VRAM_PRESET_12GB),
    ("10GB Lite FP8 Chunked", SAM_VRAM_PRESET_10GB),
    ("8GB CPU Lite Chunked", SAM_VRAM_PRESET_8GB),
)

_PRESETS: dict[str, dict[str, Any]] = {
    SAM_VRAM_PRESET_32GB: {
        "quantization": "none",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 32,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "chunk_seconds": 15.0,
        "chunk_overlap_seconds": 2.0,
    },
    SAM_VRAM_PRESET_24GB: {
        "quantization": "none",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 24,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "chunk_seconds": 10.0,
        "chunk_overlap_seconds": 1.5,
    },
    SAM_VRAM_PRESET_16GB: {
        "quantization": "fp8_scaled",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 16,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "chunk_seconds": 8.0,
        "chunk_overlap_seconds": 1.0,
    },
    SAM_VRAM_PRESET_12GB: {
        "quantization": "fp8_scaled",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 12,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "chunk_seconds": 6.0,
        "chunk_overlap_seconds": 1.0,
    },
    SAM_VRAM_PRESET_10GB: {
        "quantization": "fp8_scaled",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 8,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "chunk_seconds": 5.0,
        "chunk_overlap_seconds": 0.75,
    },
    SAM_VRAM_PRESET_8GB: {
        "quantization": "none",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 8,
        "device_mode": "cpu",
        "low_vram_lite": True,
        "chunked": True,
        "chunk_seconds": 4.0,
        "chunk_overlap_seconds": 0.5,
    },
}

_LEGACY_ALIASES = {
    "16gb_low": SAM_VRAM_PRESET_16GB,
    "fp8_scaled": SAM_VRAM_PRESET_16GB,
}


def get_sam_vram_preset(name: object) -> dict[str, Any]:
    """Return settings for a SAM-Audio VRAM preset."""

    return deepcopy(_PRESETS.get(normalize_sam_vram_preset(name), _PRESETS[SAM_VRAM_PRESET_24GB]))


def normalize_sam_vram_preset(name: object) -> str:
    """Return a supported preset name, replacing legacy auto values."""

    value = str(name or "").strip()
    value = _LEGACY_ALIASES.get(value, value)
    if value == "auto" or not value:
        return default_sam_vram_preset_name()
    if value in _PRESETS:
        return value
    return SAM_VRAM_PRESET_24GB


def select_sam_vram_preset_for_gpu(gpu_memory_gb: float) -> str:
    """Return the SAM-Audio preset that best matches detected VRAM."""

    if gpu_memory_gb >= 31.0:
        return SAM_VRAM_PRESET_32GB
    if gpu_memory_gb >= 23.0:
        return SAM_VRAM_PRESET_24GB
    if gpu_memory_gb >= 15.0:
        return SAM_VRAM_PRESET_16GB
    if gpu_memory_gb >= 11.0:
        return SAM_VRAM_PRESET_12GB
    if gpu_memory_gb >= 9.0:
        return SAM_VRAM_PRESET_10GB
    return SAM_VRAM_PRESET_8GB


def default_sam_vram_preset_name() -> str:
    """Return the runtime default SAM-Audio preset for the current GPU."""

    try:
        from acestep.gpu_config import get_global_gpu_config

        gpu_config = get_global_gpu_config()
        return select_sam_vram_preset_for_gpu(float(gpu_config.gpu_memory_gb))
    except Exception:
        return SAM_VRAM_PRESET_24GB
