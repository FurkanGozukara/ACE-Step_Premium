"""Template-guide frame alignment helpers for DiffPitcher."""

from __future__ import annotations

import math

import librosa
import numpy as np

from .diffpitcher_audio_io import pad_for_stft, trim_to_hop
from .diffpitcher_constants import (
    DIFFPITCHER_HOP,
    DIFFPITCHER_N_FFT,
    DIFFPITCHER_SAMPLE_RATE,
)

EXACT_DTW_MAX_CELLS = 3_000_000
COARSE_DTW_TARGET_CELLS = 800_000


def align_reference_f0(
    source_wav: np.ndarray,
    reference_wav: np.ndarray,
    reference_f0: np.ndarray,
    source_frames: int,
) -> tuple[np.ndarray, str]:
    """Align reference F0 values to source frames with exact or coarse DTW."""

    reference_frames = len(reference_f0)
    if source_frames <= 0 or reference_frames == 0:
        return np.zeros(max(0, source_frames), dtype=np.float32), "empty"
    source_features = _mfcc_features(source_wav)
    reference_features = _mfcc_features(reference_wav)
    if source_frames * reference_frames <= EXACT_DTW_MAX_CELLS:
        indexes = _dtw_reference_indexes(source_features, reference_features, source_frames)
        return reference_f0[indexes], "exact_dtw"
    factor = _coarse_factor(source_frames, reference_frames)
    indexes = _coarse_dtw_reference_indexes(
        source_features,
        reference_features,
        source_frames,
        reference_frames,
        factor,
    )
    return reference_f0[indexes], "coarse_dtw"


def _mfcc_features(wav: np.ndarray) -> np.ndarray:
    """Return compact frame features for vocal template alignment."""

    wav = trim_to_hop(wav)
    return librosa.feature.mfcc(
        y=pad_for_stft(wav),
        sr=DIFFPITCHER_SAMPLE_RATE,
        n_mfcc=20,
        n_fft=DIFFPITCHER_N_FFT,
        hop_length=DIFFPITCHER_HOP,
        win_length=DIFFPITCHER_N_FFT,
        center=False,
    ).astype(np.float32)


def _dtw_reference_indexes(
    source_features: np.ndarray,
    reference_features: np.ndarray,
    source_frames: int,
) -> np.ndarray:
    """Return reference frame indexes from an exact DTW path."""

    _, path = librosa.sequence.dtw(
        X=source_features,
        Y=reference_features,
        metric="euclidean",
    )
    return _reference_indexes_from_path(
        path,
        source_frames,
        reference_features.shape[-1],
    )


def _coarse_dtw_reference_indexes(
    source_features: np.ndarray,
    reference_features: np.ndarray,
    source_frames: int,
    reference_frames: int,
    factor: int,
) -> np.ndarray:
    """Return reference frame indexes from a downsampled DTW path."""

    coarse_source = _downsample_features(source_features, factor)
    coarse_reference = _downsample_features(reference_features, factor)
    coarse_indexes = _dtw_reference_indexes(
        coarse_source,
        coarse_reference,
        coarse_source.shape[-1],
    )
    source_anchors = np.arange(len(coarse_indexes), dtype=np.float32) * factor
    reference_anchors = coarse_indexes.astype(np.float32) * factor
    source_anchors = np.append(source_anchors, float(source_frames - 1))
    reference_anchors = np.append(reference_anchors, float(reference_frames - 1))
    indexes = np.interp(
        np.arange(source_frames, dtype=np.float32),
        source_anchors,
        reference_anchors,
    )
    return np.rint(np.clip(indexes, 0, reference_frames - 1)).astype(np.int64)


def _reference_indexes_from_path(
    path: np.ndarray,
    source_frames: int,
    reference_frames: int,
) -> np.ndarray:
    """Map every source frame to a reference frame from a DTW path."""

    mapping: dict[int, list[int]] = {}
    for source_index, reference_index in path[::-1]:
        mapping.setdefault(int(source_index), []).append(int(reference_index))
    indexes = np.zeros(source_frames, dtype=np.int64)
    previous_ref = 0
    for frame in range(source_frames):
        refs = mapping.get(frame)
        if refs:
            previous_ref = int(round(float(np.mean(refs))))
        indexes[frame] = min(previous_ref, reference_frames - 1)
    return indexes


def _downsample_features(features: np.ndarray, factor: int) -> np.ndarray:
    """Average feature frames by a positive integer factor."""

    if factor <= 1:
        return features
    groups = [
        features[:, start : min(start + factor, features.shape[-1])].mean(axis=1)
        for start in range(0, features.shape[-1], factor)
    ]
    return np.stack(groups, axis=1).astype(np.float32)


def _coarse_factor(source_frames: int, reference_frames: int) -> int:
    """Return a downsampling factor that bounds coarse DTW memory."""

    cells = max(1, source_frames) * max(1, reference_frames)
    return max(1, int(math.ceil(math.sqrt(cells / COARSE_DTW_TARGET_CELLS))))
