"""Upload-change handlers for Advanced generation media fields."""

from __future__ import annotations

from typing import Any

from .. import generation_handlers as gen_h
from .media_upload_preview import preview_audio_purpose_upload
from ...media_upload_values import latest_upload_path


def handle_src_audio_upload(src_audio: Any, mode: str) -> tuple[Any, Any, Any]:
    """Return preview and duration updates for Advanced source media uploads.

    Args:
        src_audio: Uploaded source audio or video path.
        mode: Current generation mode.

    Returns:
        Audio preview, video preview, and duration updates.
    """

    src_audio_path = latest_upload_path(src_audio)
    audio_preview, video_preview = preview_audio_purpose_upload(src_audio_path)
    duration_update = gen_h.handle_extract_src_audio_change(src_audio_path, mode)
    return audio_preview, video_preview, duration_update


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
