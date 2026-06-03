"""Preview helpers for Gradio media upload fields."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.audio_processing.media_io import is_video_file
from acestep.ui.gradio.media_upload_values import latest_upload_path


PREVIEW_AUDIO_SECONDS = 30.0
PREVIEW_AUDIO_TIMEOUT_SECONDS = 45.0


def preview_audio_purpose_upload(input_value: Any) -> tuple[Any, Any]:
    """Return audio and video previews for uploads consumed as audio.

    Args:
        input_value: Uploaded audio/video value from Gradio.

    Returns:
        Tuple of ``(audio_preview_update, video_preview_update)``.
    """

    input_path = latest_upload_path(input_value)
    if not input_path:
        return _hide_audio(), _hide_video()
    if not is_video_file(input_path):
        return gr.update(value=input_path, visible=True), _hide_video()

    video_update = gr.update(value=input_path, visible=True)
    try:
        audio_preview = extract_audio_preview(input_path)
    except Exception as exc:
        gr.Warning(f"Could not extract audio preview from video: {exc}")
        return _hide_audio(), video_update
    return gr.update(value=audio_preview, visible=True), video_update


def preview_video_upload(input_value: Any) -> Any:
    """Return a video preview update for a video-only upload field."""

    input_path = latest_upload_path(input_value)
    if not input_path:
        return _hide_video()
    if not is_video_file(input_path):
        gr.Warning("Upload a supported video file.")
        return _hide_video()
    return gr.update(value=input_path, visible=True)


def extract_audio_preview(input_path: str) -> str:
    """Extract a bounded temporary WAV preview from an uploaded audio or video file."""

    target_dir = Path(tempfile.mkdtemp(prefix="acestep_upload_audio_preview_"))
    target_path = target_dir / f"{Path(input_path).stem or 'upload'}_audio_preview.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-t",
        str(PREVIEW_AUDIO_SECONDS),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        str(target_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=PREVIEW_AUDIO_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg audio preview extraction timed out.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg audio preview extraction failed: {stderr}") from exc
    return str(target_path).replace("\\", "/")


def _hide_audio() -> Any:
    """Return a hidden empty audio update."""

    return gr.update(value=None, visible=False)


def _hide_video() -> Any:
    """Return a hidden empty video update."""

    return gr.update(value=None, visible=False)
