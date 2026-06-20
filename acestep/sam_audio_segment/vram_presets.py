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

SAM_VRAM_PRESET_MEASURED_PEAK_GIB: dict[str, float] = {
    # Max-of-3 measured on 2026-06-21 with Windows_Start_App.bat, browser/Gradio
    # defaults, full-song SAM-Audio, GPU0 isolated, and app restart per preset.
    SAM_VRAM_PRESET_32GB: 24.527,
    SAM_VRAM_PRESET_24GB: 17.570,
    SAM_VRAM_PRESET_16GB: 12.137,
    SAM_VRAM_PRESET_12GB: 9.348,
    SAM_VRAM_PRESET_10GB: 9.348,
}

SAM_VRAM_PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    (
        "32GB Highest Quality - 4 candidates + Judge (peak 24.5GiB)",
        SAM_VRAM_PRESET_32GB,
    ),
    (
        "24GB High Quality - 2 candidates + Judge (peak 17.6GiB)",
        SAM_VRAM_PRESET_24GB,
    ),
    (
        "16GB Normal Quality - 1 candidate (peak 12.1GiB)",
        SAM_VRAM_PRESET_16GB,
    ),
    (
        "12GB Normal Quality - 10s GPU chunks (peak 9.3GiB)",
        SAM_VRAM_PRESET_12GB,
    ),
    (
        "10GB Normal Quality - 10s GPU chunks (peak 9.3GiB)",
        SAM_VRAM_PRESET_10GB,
    ),
    ("8GB Normal Quality - CPU safe", SAM_VRAM_PRESET_8GB),
)

_PRESETS: dict[str, dict[str, Any]] = {
    SAM_VRAM_PRESET_32GB: {
        "quantization": "none",
        "attention_backend": "auto",
        "reranking_candidates": 4,
        "ranker_mode": "judge",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 32,
        "device_mode": "auto",
        "low_vram_lite": False,
        "chunked": True,
        "long_audio_mode": "chunked",
        "chunk_seconds": 20.0,
        "chunk_overlap_seconds": 5.0,
    },
    SAM_VRAM_PRESET_24GB: {
        "quantization": "none",
        "attention_backend": "auto",
        "reranking_candidates": 2,
        "ranker_mode": "judge",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 32,
        "device_mode": "auto",
        "low_vram_lite": False,
        "chunked": True,
        "long_audio_mode": "chunked",
        "chunk_seconds": 20.0,
        "chunk_overlap_seconds": 5.0,
    },
    SAM_VRAM_PRESET_16GB: {
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
        "long_audio_mode": "chunked",
        "chunk_seconds": 20.0,
        "chunk_overlap_seconds": 5.0,
    },
    SAM_VRAM_PRESET_12GB: {
        "quantization": "fp8_scaled",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 32,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "long_audio_mode": "chunked",
        "chunk_seconds": 10.0,
        "chunk_overlap_seconds": 5.0,
    },
    SAM_VRAM_PRESET_10GB: {
        "quantization": "fp8_scaled",
        "attention_backend": "auto",
        "reranking_candidates": 1,
        "ranker_mode": "none",
        "predict_spans": False,
        "subprocess": True,
        "ode_steps": 32,
        "device_mode": "auto",
        "low_vram_lite": True,
        "chunked": True,
        "long_audio_mode": "chunked",
        "chunk_seconds": 10.0,
        "chunk_overlap_seconds": 5.0,
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
        "long_audio_mode": "chunked",
        "chunk_seconds": 20.0,
        "chunk_overlap_seconds": 5.0,
    },
}

_LEGACY_ALIASES = {
    "16gb_low": SAM_VRAM_PRESET_16GB,
    "fp8_scaled": SAM_VRAM_PRESET_16GB,
}


def get_sam_vram_preset(name: object) -> dict[str, Any]:
    """Return settings for a SAM-Audio VRAM preset."""

    return deepcopy(
        _PRESETS.get(normalize_sam_vram_preset(name), _PRESETS[SAM_VRAM_PRESET_24GB])
    )


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

    if gpu_memory_gb >= 28.0:
        return SAM_VRAM_PRESET_32GB
    if gpu_memory_gb >= 20.0:
        return SAM_VRAM_PRESET_24GB
    if gpu_memory_gb >= 14.0:
        return SAM_VRAM_PRESET_16GB
    if gpu_memory_gb >= 11.0:
        return SAM_VRAM_PRESET_12GB
    if gpu_memory_gb >= 10.0:
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
