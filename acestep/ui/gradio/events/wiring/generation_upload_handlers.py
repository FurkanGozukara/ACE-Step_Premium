"""Upload-change handlers for Advanced generation media fields."""

from __future__ import annotations

from typing import Any

import gradio as gr

from .. import generation_handlers as gen_h
from .media_upload_preview import (
    preview_audio_purpose_upload,
    preview_audio_purpose_upload_direct,
)
from .media_range_preview import preview_source_range
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


def use_generated_result_as_source(
    generated_audio: Any,
    mode: str,
    repainting_start: Any = 0.0,
    repainting_end: Any = -1,
) -> tuple[Any, Any, Any, Any, str | None, Any, Any, Any, Any]:
    """Use the latest inline generated result as the Advanced source audio.

    Args:
        generated_audio: Latest generated audio value from the inline preview.
        mode: Current generation mode, used for source-locked duration updates.
        repainting_start: Current source segment start in seconds.
        repainting_end: Current source segment end in seconds, or ``-1`` for source end.

    Returns:
        Source upload, audio preview, video preview, duration, original preview
        state, Remix source start, Remix source end, selected-range audio preview,
        and selected-range video preview updates.
    """

    generated_path = latest_upload_path(generated_audio)
    if not generated_path:
        gr.Warning("Generate a result first, then use it as Source Audio.")
        return (gr.skip(),) * 9
    audio_preview, video_preview = preview_audio_purpose_upload_direct(generated_path)
    duration_update = gen_h.handle_extract_src_audio_change(generated_path, mode)
    range_start, range_end, source_start_update, source_end_update = (
        _generated_source_range_values(mode, repainting_start, repainting_end)
    )
    range_audio_preview, range_video_preview = preview_source_range(
        generated_path,
        _update_value_path(audio_preview),
        _update_value_path(audio_preview),
        range_start,
        range_end,
        mode,
    )
    gr.Info("Generated result is now the Source Audio.")
    return (
        generated_path,
        audio_preview,
        video_preview,
        duration_update,
        _update_value_path(audio_preview),
        source_start_update,
        source_end_update,
        range_audio_preview,
        range_video_preview,
    )


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


def _generated_source_range_values(mode: Any, start: Any, end: Any) -> tuple[Any, Any, Any, Any]:
    """Return effective range values plus Gradio control updates for source copy."""

    if str(mode or "").strip().lower() != "remix":
        return start, end, gr.skip(), gr.skip()
    return 0.0, -1, gr.update(value=0.0), gr.update(value=-1)


def _report_progress(progress: Any | None, value: float, desc: str) -> None:
    """Report progress when called from a Gradio event context."""

    if progress is not None:
        progress(value, desc=desc)
