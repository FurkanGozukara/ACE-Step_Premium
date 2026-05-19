"""Audio loading helpers for dataset tensor preprocessing."""

from __future__ import annotations

import subprocess

import numpy as np
import torch
import torchaudio
from loguru import logger


def load_audio_stereo(
    audio_path: str,
    target_sample_rate: int,
    max_duration: float,
) -> tuple[torch.Tensor, int]:
    """Load audio, resample to the target rate, convert to stereo, and truncate."""

    loaded = _load_with_soundfile(audio_path)
    if loaded is None:
        loaded = _load_with_ffmpeg(audio_path, target_sample_rate, max_duration)
    if loaded is None:
        raise RuntimeError(f"Could not load audio for preprocessing: {audio_path}")

    audio, sample_rate = loaded
    audio = _ensure_stereo(audio)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        audio = resampler(audio)

    return _trim_audio(audio, target_sample_rate, max_duration), target_sample_rate


def _load_with_soundfile(audio_path: str) -> tuple[torch.Tensor, int] | None:
    """Load audio through libsndfile when the file format is supported."""

    try:
        import soundfile as sf

        data, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    except Exception as exc:
        logger.debug(f"soundfile preprocessing load failed for {audio_path}: {exc}")
        return None

    return torch.from_numpy(data.T.copy()), int(sample_rate)


def _load_with_ffmpeg(
    audio_path: str,
    target_sample_rate: int,
    max_duration: float,
) -> tuple[torch.Tensor, int] | None:
    """Load audio through the standard FFmpeg executable."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(target_sample_rate),
    ]
    if max_duration > 0:
        command.extend(["-t", str(max_duration)])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])

    try:
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        logger.debug(f"ffmpeg preprocessing load failed for {audio_path}: {exc}")
        return None

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = stderr[0] if stderr else f"exit code {result.returncode}"
        logger.debug(f"ffmpeg preprocessing load failed for {audio_path}: {detail}")
        return None

    data = np.frombuffer(result.stdout, dtype=np.float32)
    if data.size < 2:
        logger.debug(f"ffmpeg preprocessing load returned no audio for {audio_path}")
        return None
    if data.size % 2:
        data = data[:-1]

    audio = torch.from_numpy(data.reshape(-1, 2).T.copy())
    return audio, target_sample_rate


def _ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    """Return a two-channel audio tensor."""

    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    if audio.shape[0] > 2:
        return audio[:2, :]
    return audio


def _trim_audio(audio: torch.Tensor, target_sample_rate: int, max_duration: float) -> torch.Tensor:
    """Trim audio to the requested maximum duration."""

    max_samples = int(max_duration * target_sample_rate)
    if max_samples >= 0 and audio.shape[1] > max_samples:
        return audio[:, :max_samples]
    return audio
