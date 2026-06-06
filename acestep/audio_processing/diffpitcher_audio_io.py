"""Audio loading and frame-shape helpers for DiffPitcher."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .diffpitcher_constants import (
    DIFFPITCHER_HOP,
    DIFFPITCHER_SAMPLE_RATE,
    DIFFPITCHER_STFT_PAD,
)
from .media_io import is_video_file, read_media_audio


def to_mono_24k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return mono 24 kHz float32 audio for DiffPitcher."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError("audio must be mono or channel-last stereo.")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, -1.0, 1.0)
    if sample_rate != DIFFPITCHER_SAMPLE_RATE:
        array = librosa.resample(
            array,
            orig_sr=sample_rate,
            target_sr=DIFFPITCHER_SAMPLE_RATE,
        )
    return np.asarray(array, dtype=np.float32)


def load_mono_24k(path: str | Path) -> np.ndarray:
    """Load an audio file as mono 24 kHz float32 audio."""

    source = Path(path).expanduser().resolve()
    if is_video_file(source):
        audio, sample_rate = read_media_audio(source)
        return to_mono_24k(audio, sample_rate)
    try:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
    except (OSError, RuntimeError):
        try:
            loaded, sample_rate = librosa.load(str(source), sr=None, mono=False)
            audio = loaded.T if loaded.ndim == 2 else loaded
        except Exception:
            audio, sample_rate = read_media_audio(source)
    return to_mono_24k(np.asarray(audio, dtype=np.float32), int(sample_rate))


def restore_output_shape(
    mono_audio: np.ndarray,
    original_samples: int,
    original_channels: int,
    sample_rate: int,
) -> np.ndarray:
    """Resample DiffPitcher output and restore the original channel count."""

    output = np.asarray(mono_audio, dtype=np.float32)
    if sample_rate != DIFFPITCHER_SAMPLE_RATE:
        output = librosa.resample(
            output,
            orig_sr=DIFFPITCHER_SAMPLE_RATE,
            target_sr=sample_rate,
        )
    output = match_length(output, original_samples)
    if original_channels <= 1:
        return output.astype(np.float32, copy=False)
    return np.repeat(output[:, None], original_channels, axis=1).astype(np.float32)


def trim_to_hop(wav: np.ndarray) -> np.ndarray:
    """Trim or pad audio to a positive hop-aligned length."""

    audio = np.asarray(wav, dtype=np.float32)
    if len(audio) < DIFFPITCHER_HOP:
        return match_length(audio, DIFFPITCHER_HOP)
    length = max(DIFFPITCHER_HOP, (len(audio) // DIFFPITCHER_HOP) * DIFFPITCHER_HOP)
    return audio[:length]


def frame_count(wav: np.ndarray) -> int:
    """Return the unpadded DiffPitcher frame count for audio."""

    return max(1, len(trim_to_hop(wav)) // DIFFPITCHER_HOP)


def pad_count(count: int) -> int:
    """Return a frame count padded to DiffPitcher's multiple-of-eight requirement."""

    remainder = count % 8
    return count if remainder == 0 else count + (8 - remainder)


def pad_for_stft(wav: np.ndarray) -> np.ndarray:
    """Pad waveform using DiffPitcher's STFT convention."""

    mode = "reflect" if len(wav) > 1 else "constant"
    return np.pad(wav, DIFFPITCHER_STFT_PAD, mode=mode)


def match_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Trim or zero-pad audio to an exact sample count."""

    target = max(0, int(target_samples))
    array = np.asarray(audio, dtype=np.float32)
    if len(array) >= target:
        return array[:target]
    return np.pad(array, (0, target - len(array)), mode="constant")
