"""Output helpers for the simple Create tab media preview."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from loguru import logger

from .simple_run_paths import resolve_simple_audio_path
from .simple_video_artifacts import export_simple_video_artifacts


def clear_simple_media_preview() -> tuple[Any, Any]:
    """Clear the simple-tab audio/video preview before a new generation."""

    return gr.update(value=None, visible=True), gr.update(value=None, visible=False)


def build_simple_media_preview(
    audio_path: Any,
    status: str | None,
    image_path: Any,
    video_resolution: str | None,
    generated_files: Any = None,
) -> Iterator[tuple[Any, Any, str]]:
    """Yield final simple-tab media preview updates.

    If an image is provided, an MP4 is generated from the first audio result and
    the uploaded image. Otherwise the audio player stays visible.
    """

    normalized_audio = _normalize_path(audio_path)
    if not normalized_audio:
        yield gr.update(value=None, visible=True), gr.update(value=None, visible=False), (
            status or "Generation finished without an audio output."
        )
        return
    output_audio = resolve_simple_audio_path(normalized_audio, generated_files)

    normalized_image = _normalize_path(image_path)
    if not normalized_image:
        yield (
            gr.update(value=output_audio, visible=True),
            gr.update(value=None, visible=False),
            _final_status(status, output_audio, "Audio ready."),
        )
        return

    yield (
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=True),
        f"Creating MP4 video from uploaded image...\nResolution: {video_resolution or '1080p'}",
    )

    try:
        artifacts = export_simple_video_artifacts(
            output_audio,
            normalized_image,
            video_resolution,
        )
    except (OSError, RuntimeError) as exc:
        logger.exception("Failed to create simple-tab MP4 preview")
        yield (
            gr.update(value=output_audio, visible=True),
            gr.update(value=None, visible=False),
            f"{status or 'Generation complete.'}\nMP4 creation failed: {exc}",
        )
        return

    yield (
        gr.update(value=None, visible=False),
        gr.update(value=artifacts.video_path, visible=True),
        _final_status(
            status,
            artifacts.video_path,
            "MP4 video ready.",
            image_path=artifacts.image_path,
        ),
    )


def _normalize_path(value: Any) -> str:
    """Return a normalized path string from Gradio file/image values."""

    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    elif hasattr(value, "path"):
        value = getattr(value, "path")
    elif hasattr(value, "name"):
        value = getattr(value, "name")
    return str(value or "").strip().replace("\\", "/")


def _final_status(
    status: str | None,
    media_path: str,
    ready_message: str,
    *,
    image_path: str | None = None,
) -> str:
    """Return compact final status with the saved run folder."""

    if status and "failed" in status.lower():
        return status.strip()

    run_folder = Path(media_path).expanduser().parent
    if image_path:
        return (
            f"{ready_message} Outputs are saved.\n"
            f"MP4: {media_path}\n"
            f"Image: {image_path}\n"
            f"Folder: {run_folder}"
        )
    return f"{ready_message}\nGeneration complete. Outputs are saved.\nFolder: {run_folder}"
