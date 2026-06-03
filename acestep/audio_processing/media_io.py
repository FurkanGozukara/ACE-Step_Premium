"""Media IO helpers for ACE-Step audio processing."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from acestep.audio_utils import save_audio
from .media_duration import probe_media_duration_seconds


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
SUPPORTED_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def is_supported_media(path: str | Path) -> bool:
    """Return whether a path has a supported audio or video extension."""

    return Path(path).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def is_video_file(path: str | Path) -> bool:
    """Return whether a path looks like a supported video file."""

    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def read_media_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio from an audio or video file as float32 channel-last samples.

    Args:
        path: Audio or video file path.

    Returns:
        Tuple of audio array and sample rate.

    Raises:
        FileNotFoundError: If the source file does not exist.
        RuntimeError: If decoding fails.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input media file not found: {source}")
    if not is_video_file(source):
        try:
            audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
            return audio, int(sample_rate)
        except Exception:
            pass
    return _read_with_ffmpeg(source)


def media_audio_duration_seconds(path: str | Path) -> float:
    """Return media audio duration from metadata without decoding full video files."""

    source = Path(path).expanduser().resolve()
    if not is_video_file(source):
        try:
            return float(sf.info(str(source)).duration)
        except Exception:
            pass
    return probe_media_duration_seconds(source)


def save_processed_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    output_format: str,
) -> str:
    """Save processed channel-last audio in a supported output format."""

    target_format = _normalize_output_format(output_format)
    channel_first = _to_channel_first(audio)
    return save_audio(
        channel_first,
        output_path,
        sample_rate=sample_rate,
        format=target_format,
        channels_first=True,
        mp3_bitrate="320k",
        mp3_sample_rate=48000,
    ).replace("\\", "/")


def mux_video_with_audio(
    source_video: str | Path,
    processed_audio: str | Path,
    output_video: str | Path,
) -> str:
    """Copy the source video stream and replace its audio track."""

    source = Path(source_video).expanduser().resolve()
    audio = Path(processed_audio).expanduser().resolve()
    target = Path(output_video).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(target),
    ]
    _run_ffmpeg(cmd, "ffmpeg video mux failed")
    return str(target).replace("\\", "/")


def _read_with_ffmpeg(source: Path) -> tuple[np.ndarray, int]:
    """Decode media audio with ffmpeg into a temporary WAV file."""

    with tempfile.TemporaryDirectory(prefix="acestep_audio_decode_") as temp_dir:
        wav_path = Path(temp_dir) / "decoded.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_f32le",
            str(wav_path),
        ]
        _run_ffmpeg(cmd, "ffmpeg media decode failed")
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        return audio, int(sample_rate)


def _run_ffmpeg(cmd: list[str], message: str) -> None:
    """Run ffmpeg and raise a compact error when it fails."""

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{message}: timed out.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"{message}: {stderr}") from exc


def _to_channel_first(audio: np.ndarray) -> np.ndarray:
    """Return audio in channel-first layout for the shared AudioSaver."""

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    return arr.T


def _normalize_output_format(value: Any) -> str:
    """Return a supported output format for processed media."""

    normalized = str(value or "wav").strip().lower()
    return normalized if normalized in {"wav", "flac", "mp3"} else "wav"
