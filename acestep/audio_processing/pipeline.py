"""Orchestrate the ACE-Step audio processing chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dsp_enhancement import (
    ambience_shaping,
    harmonic_enrichment,
    hf_refinement,
    stereo_depth,
    stereo_width,
    timing_humanizer,
)
from .dsp_mastering import (
    glue_compressor,
    lufs_normalize,
    measure_lufs,
    midside_eq,
    multiband_compressor,
    soft_clipper,
    tape_saturation,
)
from .presets import STAGE_KEYS, STAGE_LABELS
from .settings import AudioProcessingSettings


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ProcessedAudio:
    """Processed audio plus metering information.

    Args:
        before: Input audio used for processing, channel-last.
        after: Processed audio, channel-last.
        sample_rate: Audio sample rate in Hz.
        lufs_before: Integrated loudness before processing.
        lufs_after: Integrated loudness after processing.
        duration_seconds: Processed duration in seconds.

    Returns:
        Immutable processing result.
    """

    before: np.ndarray
    after: np.ndarray
    sample_rate: int
    lufs_before: float
    lufs_after: float
    duration_seconds: float


def process_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    settings: AudioProcessingSettings,
    *,
    max_seconds: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ProcessedAudio:
    """Run the configured processing chain on a NumPy audio array.

    Args:
        audio: Input audio as mono or channel-last float array.
        sample_rate: Sample rate in Hz.
        settings: Processing settings and enabled stages.
        max_seconds: Optional preview duration limit.
        progress_callback: Optional callback receiving progress fraction and label.

    Returns:
        Processed audio and before/after LUFS metadata.
    """

    working = _prepare_audio(audio, sample_rate, max_seconds)
    before = working.copy()
    lufs_before = measure_lufs(before, sample_rate)
    steps = _stage_steps(sample_rate, settings)
    for index, (key, label, function) in enumerate(steps, start=1):
        if settings.stage_enabled(key):
            working = function(working)
            working = _sanitize_audio(working)
        if progress_callback:
            progress_callback(index / len(steps), label)
    lufs_after = measure_lufs(working, sample_rate)
    return ProcessedAudio(
        before=before,
        after=working.astype(np.float32, copy=False),
        sample_rate=int(sample_rate),
        lufs_before=lufs_before,
        lufs_after=lufs_after,
        duration_seconds=len(working) / float(sample_rate),
    )


def _stage_steps(
    sample_rate: int,
    settings: AudioProcessingSettings,
) -> list[tuple[str, str, Callable[[np.ndarray], np.ndarray]]]:
    """Return ordered processing stages for the current settings."""

    return [
        ("phase", STAGE_LABELS["phase"], lambda audio: stereo_depth(
            audio, sample_rate, settings.stage_value("phase")
        )),
        ("stereo", STAGE_LABELS["stereo"], lambda audio: stereo_width(
            audio, settings.stage_value("stereo")
        )),
        ("hf", STAGE_LABELS["hf"], lambda audio: hf_refinement(
            audio, sample_rate, settings.stage_value("hf")
        )),
        ("harmonic", STAGE_LABELS["harmonic"], lambda audio: harmonic_enrichment(
            audio, settings.stage_value("harmonic")
        )),
        ("jitter", STAGE_LABELS["jitter"], lambda audio: timing_humanizer(
            audio, sample_rate, settings.stage_value("jitter")
        )),
        ("noise", STAGE_LABELS["noise"], lambda audio: ambience_shaping(
            audio, sample_rate, settings.stage_value("noise")
        )),
        ("multiband", STAGE_LABELS["multiband"], lambda audio: multiband_compressor(
            audio, sample_rate, settings.stage_value("multiband")
        )),
        ("tape", STAGE_LABELS["tape"], lambda audio: tape_saturation(
            audio, sample_rate, settings.stage_value("tape")
        )),
        ("glue", STAGE_LABELS["glue"], lambda audio: glue_compressor(
            audio, sample_rate, settings.stage_value("glue")
        )),
        ("mseq", STAGE_LABELS["mseq"], lambda audio: midside_eq(
            audio, sample_rate, settings.stage_value("mseq")
        )),
        ("clip", STAGE_LABELS["clip"], lambda audio: soft_clipper(
            audio, settings.stage_value("clip")
        )),
        ("lufs", STAGE_LABELS["lufs"], lambda audio: lufs_normalize(
            audio, sample_rate, settings.stage_value("lufs")
        )),
    ]


def _prepare_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_seconds: float | None,
) -> np.ndarray:
    """Return float32 stereo channel-last audio ready for processing."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    working = np.asarray(audio, dtype=np.float32)
    if working.ndim == 1:
        working = np.stack([working, working], axis=1)
    if working.ndim != 2:
        raise ValueError("audio must be a mono or channel-last 2D array.")
    if working.shape[1] == 1:
        working = np.repeat(working, 2, axis=1)
    if max_seconds is not None and max_seconds > 0:
        working = working[: int(sample_rate * float(max_seconds))]
    return _sanitize_audio(working)


def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
    """Remove invalid samples and keep the signal inside a safe range."""

    sanitized = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(sanitized, -1.0, 1.0).astype(np.float32, copy=False)


def active_stage_labels(settings: AudioProcessingSettings) -> list[str]:
    """Return display labels for stages currently enabled."""

    return [STAGE_LABELS[key] for key in STAGE_KEYS if settings.stage_enabled(key)]
