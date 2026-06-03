"""Upload-change handlers for Advanced generation media fields."""

from __future__ import annotations

from typing import Any

import gradio as gr

from .. import generation_handlers as gen_h
from .media_upload_preview import (
    preview_audio_purpose_upload,
    preview_audio_purpose_upload_direct,
)
from ...media_upload_values import latest_upload_path, resolve_effective_source_audio


def handle_src_audio_upload(src_audio: Any, mode: str) -> tuple[Any, Any, Any, str | None]:
    """Return immediate preview updates for Advanced source media uploads.

    Args:
        src_audio: Uploaded source audio or video path.
        mode: Current generation mode; accepted to keep the event signature stable.

    Returns:
        Audio preview, video preview, no-op duration update, and original preview path.
    """

    src_audio_path = latest_upload_path(src_audio)
    audio_preview, video_preview = preview_audio_purpose_upload_direct(src_audio_path)
    return audio_preview, video_preview, gr.update(), _update_value_path(audio_preview)


def finalize_src_audio_upload(
    src_audio: Any,
    mode: str,
    progress=gr.Progress(),
) -> tuple[Any, Any, Any, str | None]:
    """Return extracted preview and duration updates after the fast preview is shown.

    Args:
        src_audio: Uploaded source audio or video path.
        mode: Current generation mode.
        progress: Optional Gradio progress reporter.

    Returns:
        Audio preview, video preview, duration update, and original preview path.
    """

    src_audio_path = latest_upload_path(src_audio)
    _report_progress(progress, 0.05, "Preparing source preview")
    audio_preview, video_preview = preview_audio_purpose_upload(
        src_audio_path,
        progress=progress,
    )
    _report_progress(progress, 0.9, "Reading source duration")
    duration_update = gen_h.handle_extract_src_audio_change(src_audio_path, mode)
    _report_progress(progress, 1.0, "Source preview ready")
    return audio_preview, video_preview, duration_update, _update_value_path(audio_preview)


def handle_src_audio_preview_change(
    src_audio: Any,
    src_audio_preview: Any,
    src_audio_preview_original: Any,
    mode: str,
) -> Any:
    """Return duration update for an edited Source Audio Preview.

    Args:
        src_audio: Original Source Audio upload value.
        src_audio_preview: Current Source Audio Preview value.
        src_audio_preview_original: Preview path created from the original upload.
        mode: Current generation mode.

    Returns:
        A duration update for Extract/Lego modes.
    """

    effective_source = resolve_effective_source_audio(
        src_audio,
        src_audio_preview,
        src_audio_preview_original,
    )
    return gen_h.handle_extract_src_audio_change(effective_source, mode)


def handle_reference_media_upload(dit_handler: Any, *args: Any) -> tuple[Any, Any, Any]:
    """Return instruction and preview updates for Advanced reference uploads.

    Args:
        dit_handler: Generation handler used by instruction rendering.
        *args: Instruction inputs ending with the reference media path.

    Returns:
        Instruction, audio preview, and video preview updates.
    """

    instruction_update = gen_h.update_instruction_ui(dit_handler, *args)
    reference_audio = latest_upload_path(args[4] if len(args) >= 5 else None)
    audio_preview, video_preview = preview_audio_purpose_upload(reference_audio)
    return instruction_update, audio_preview, video_preview


def _update_value_path(update: Any) -> str | None:
    """Return the filepath embedded in a Gradio update payload."""

    if isinstance(update, dict):
        return latest_upload_path(update.get("value"))
    return latest_upload_path(update)


def _report_progress(progress: Any | None, value: float, desc: str) -> None:
    """Report progress when called from a Gradio event context."""

    if progress is not None:
        progress(value, desc=desc)
