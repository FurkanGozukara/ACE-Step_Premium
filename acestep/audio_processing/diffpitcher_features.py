"""Compatibility imports for DiffPitcher feature helpers."""

from __future__ import annotations

from .diffpitcher_audio_io import load_mono_24k, restore_output_shape, to_mono_24k
from .diffpitcher_constants import (
    DIFFPITCHER_F0_BIN,
    DIFFPITCHER_F0_CLAMP_MAX,
    DIFFPITCHER_F0_MAX,
    DIFFPITCHER_F0_MIN,
    DIFFPITCHER_HOP,
    DIFFPITCHER_MEL_MAX,
    DIFFPITCHER_MEL_MIN,
    DIFFPITCHER_N_FFT,
    DIFFPITCHER_N_MELS,
    DIFFPITCHER_SAMPLE_RATE,
)
from .diffpitcher_mel_features import log_mel_from_wav, world_mel_from_wav
from .diffpitcher_pitch_features import (
    estimate_f0_world,
    log_f0_bins,
    matched_reference_f0,
    midi_f0_to_frames,
)

__all__ = [
    "DIFFPITCHER_F0_BIN",
    "DIFFPITCHER_F0_CLAMP_MAX",
    "DIFFPITCHER_F0_MAX",
    "DIFFPITCHER_F0_MIN",
    "DIFFPITCHER_HOP",
    "DIFFPITCHER_MEL_MAX",
    "DIFFPITCHER_MEL_MIN",
    "DIFFPITCHER_N_FFT",
    "DIFFPITCHER_N_MELS",
    "DIFFPITCHER_SAMPLE_RATE",
    "estimate_f0_world",
    "load_mono_24k",
    "log_f0_bins",
    "log_mel_from_wav",
    "matched_reference_f0",
    "midi_f0_to_frames",
    "restore_output_shape",
    "to_mono_24k",
    "world_mel_from_wav",
]
