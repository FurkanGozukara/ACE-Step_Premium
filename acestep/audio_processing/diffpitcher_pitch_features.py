"""Pitch-guide extraction helpers for DiffPitcher."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pyworld as pw

from .diffpitcher_alignment import align_reference_f0
from .diffpitcher_audio_io import (
    frame_count,
    match_length,
    pad_count,
    trim_to_hop,
)
from .diffpitcher_constants import (
    DIFFPITCHER_F0_BIN,
    DIFFPITCHER_F0_CLAMP_MAX,
    DIFFPITCHER_F0_MAX,
    DIFFPITCHER_F0_MIN,
    DIFFPITCHER_HOP,
    DIFFPITCHER_SAMPLE_RATE,
)
from .diffpitcher_mel_features import world_float64
from .process_logging import ProcessCallback, emit_process_message


def matched_reference_f0(
    source_wav: np.ndarray,
    reference_wav: np.ndarray,
    *,
    progress_callback: ProcessCallback | None = None,
) -> np.ndarray:
    """Return reference F0 aligned to the source frame count."""

    source_frames = frame_count(source_wav)
    reference_f0 = estimate_f0_world(reference_wav, padding=False)
    if len(reference_f0) == 0:
        return np.zeros(pad_count(source_frames), dtype=np.float32)
    aligned, mode = align_reference_f0(
        source_wav,
        reference_wav,
        reference_f0,
        source_frames,
    )
    emit_process_message(
        progress_callback,
        (
            "DiffPitcher pitch fix: reference alignment "
            f"{mode} ({source_frames} source frames, {len(reference_f0)} reference frames)"
        ),
    )
    return _pad_1d(aligned)


def estimate_f0_world(wav: np.ndarray, *, padding: bool = True) -> np.ndarray:
    """Estimate F0 using WORLD and optional DiffPitcher frame padding."""

    wav = trim_to_hop(wav)
    wav64 = world_float64(wav)
    f0, time_axis = pw.dio(
        wav64,
        fs=DIFFPITCHER_SAMPLE_RATE,
        frame_period=DIFFPITCHER_HOP / DIFFPITCHER_SAMPLE_RATE * 1000,
        f0_floor=DIFFPITCHER_F0_MIN,
        f0_ceil=DIFFPITCHER_F0_CLAMP_MAX,
    )
    f0 = pw.stonemask(wav64, f0, time_axis, DIFFPITCHER_SAMPLE_RATE)
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    f0 = f0[: frame_count(wav)]
    return _pad_1d(f0) if padding else f0


def log_f0_bins(f0: np.ndarray) -> np.ndarray:
    """Convert Hz F0 values to DiffPitcher log-frequency bins."""

    values = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0)
    bins = np.zeros_like(values, dtype=np.int64)
    voiced = values > 0
    if not np.any(voiced):
        return bins
    f0_mel = 12.0 * np.log2(values[voiced] / DIFFPITCHER_F0_MIN) + 1.0
    f0_mel_min = 1.0
    f0_mel_max = 12.0 * np.log2(DIFFPITCHER_F0_MAX / DIFFPITCHER_F0_MIN) + 1.0
    scaled = (f0_mel - f0_mel_min) * (DIFFPITCHER_F0_BIN - 2)
    scaled = scaled / (f0_mel_max - f0_mel_min) + 1.0
    bins[voiced] = np.rint(np.clip(scaled, 1, DIFFPITCHER_F0_BIN - 1)).astype(np.int64)
    return bins


def midi_f0_to_frames(midi_path: str | Path, target_frames: int) -> np.ndarray:
    """Convert a MIDI file into frame-aligned target F0 values."""

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    roll = midi.get_piano_roll(fs=100)
    roll = np.pad(roll, ((0, 0), (0, 1000)), constant_values=0)
    onsets = np.round(midi.get_onsets() * 100 - 1).astype(int)
    onsets = onsets[(onsets >= 0) & (onsets < roll.shape[1])]
    if len(onsets):
        roll[:, onsets] = 0
    midi_hz = np.zeros(roll.shape[1], dtype=np.float32)
    for frame in range(roll.shape[1]):
        notes = np.flatnonzero(roll[:, frame] > 0)
        if len(notes):
            midi_hz[frame] = float(librosa.midi_to_hz(notes[0]))
    return _nearest_align(midi_hz, target_frames)


def _pad_1d(values: np.ndarray) -> np.ndarray:
    """Pad a vector to a multiple of eight frames."""

    return match_length(values, pad_count(len(values)))


def _nearest_align(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Nearest-neighbor align frame values to a target frame count."""

    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros(target_frames, dtype=np.float32)
    if len(values) == target_frames:
        return values
    indexes = np.round(np.linspace(0, len(values) - 1, target_frames)).astype(int)
    return values[np.clip(indexes, 0, len(values) - 1)].astype(np.float32)
