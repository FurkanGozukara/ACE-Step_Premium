"""Input path selection for Audio Processing UI handlers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from acestep.ui.gradio.media_upload_values import latest_upload_path


def effective_single_file_input(
    input_value: Any,
    audio_preview_value: Any,
    local_path_value: Any = None,
) -> str | None:
    """Return an explicit local path, edited audio preview, or upload value."""

    local_path = local_media_path(local_path_value)
    if local_path:
        return local_path
    return latest_upload_path(audio_preview_value) or latest_upload_path(input_value)


def workflow_source_input(
    input_value: Any,
    audio_preview_value: Any,
    local_path_value: Any = None,
) -> str | None:
    """Return the media path Auto-Editor should analyze for workflow export."""

    return (
        latest_upload_path(audio_preview_value)
        or latest_upload_path(input_value)
        or local_media_path(local_path_value)
    )


def workflow_media_reference(source_path: str, local_path_value: Any = None) -> str:
    """Return the media path to embed in exported editor workflows."""

    return local_media_path(local_path_value) or source_path


def local_media_path(value: Any) -> str | None:
    """Return a user-entered local media path, or ``None`` when blank."""

    path = str(value or "").strip().strip("\"'")
    return path or None


def workflow_reference_note(
    source_path: str,
    media_reference: str,
    local_path_value: Any = None,
) -> str | None:
    """Return a UI note about workflow media-reference accuracy."""

    if local_media_path(local_path_value):
        return f"Media reference: `{media_reference}`"
    if not is_gradio_temp_upload(source_path):
        return None
    return (
        "Media reference: Gradio temp upload. Fill Local Audio/Video Path to make "
        "the editor project point at the original media file."
    )


def is_gradio_temp_upload(path: str) -> bool:
    """Return whether a path points inside Gradio's temporary upload folder."""

    try:
        resolved = Path(path).expanduser().resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
    except (OSError, RuntimeError):
        return False
    if not resolved.is_relative_to(temp_root):
        return False
    return any(part.casefold() == "gradio" for part in resolved.parts)
