"""Upload preview helpers for the Audio Processing page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from acestep.audio_processing.media_io import is_video_file
from acestep.ui.gradio.media_upload_values import latest_upload_path


def preview_upload(input_value: Any, disable_preview: Any = False) -> tuple[Any, Any, str]:
    """Return audio/video preview updates for an uploaded Audio Processing file.

    Args:
        input_value: Gradio file upload value.
        disable_preview: When true, hide previews so Gradio does not post-process media.

    Returns:
        Audio preview update, video preview update, and status markdown.
    """

    input_path = latest_upload_path(input_value)
    if not input_path:
        return _hide_audio(), _hide_video(), "Upload an audio or video file first."
    if bool(disable_preview):
        return (
            _hide_audio(),
            _hide_video(),
            (
                "Upload preview disabled. Processing will use original file: "
                f"`{Path(input_path).name}`"
            ),
        )
    if is_video_file(input_path):
        return (
            _hide_audio(),
            gr.update(value=input_path, visible=True),
            f"Loaded video: `{Path(input_path).name}`",
        )
    return (
        gr.update(value=input_path, visible=True),
        _hide_video(),
        f"Loaded audio: `{Path(input_path).name}`",
    )


def preview_diffpitcher_reference(input_value: Any) -> tuple[Any, Any, str]:
    """Return audio/video preview updates for the DiffPitcher reference guide."""

    audio_update, video_update, status = preview_upload(input_value)
    if status.startswith("Upload"):
        status = "Select a reference vocal audio or video file for template mode."
    elif status.startswith("Loaded video"):
        status = status.replace("Loaded video", "Loaded reference video", 1)
    elif status.startswith("Loaded audio"):
        status = status.replace("Loaded audio", "Loaded reference audio", 1)
    return audio_update, video_update, status


def _hide_audio() -> Any:
    """Return a hidden empty audio preview update."""

    return gr.update(value=None, visible=False)


def _hide_video() -> Any:
    """Return a hidden empty video preview update."""

    return gr.update(value=None, visible=False)
