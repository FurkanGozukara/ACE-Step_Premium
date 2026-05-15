"""Output helpers for the simple Create tab media preview."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from loguru import logger

from .simple_run_paths import resolve_simple_audio_path
from .simple_video_audio_paths import audio_paths_for_video_export, dedupe_paths
from .simple_video_artifacts import SimpleVideoArtifacts, export_simple_video_artifacts


def clear_simple_media_preview() -> tuple[Any, Any]:
    """Clear the simple-tab audio/video preview before a new generation."""

    return gr.update(value=None, visible=True), gr.update(value=None, visible=False)


def clear_simple_generated_files() -> Any:
    """Clear the simple-tab all-files list before a new generation."""

    return gr.update(value=None, visible=False)


def build_simple_generated_files_update(generated_files: Any) -> Any:
    """Show generated run files in the simple tab when files are available."""

    files = _flatten_paths(generated_files)
    if not files:
        return gr.update(value=None, visible=False)
    return gr.update(value=files, visible=True)


def build_simple_media_preview(
    audio_path: Any,
    status: str | None,
    image_path: Any,
    video_resolution: str | None,
    generated_files: Any = None,
) -> Iterator[tuple[Any, Any, str, Any]]:
    """Yield final simple-tab media preview updates.

    If an image is provided, MP4s are generated for each song in the run while
    previewing the first video. Otherwise the audio player stays visible.
    """

    generated_paths = _flatten_paths(generated_files)
    generated_files_update = build_simple_generated_files_update(generated_paths)
    normalized_audio = _normalize_path(audio_path)
    if not normalized_audio:
        yield (
            gr.update(value=None, visible=True),
            gr.update(value=None, visible=False),
            status or "Generation finished without an audio output.",
            generated_files_update,
        )
        return
    output_audio = resolve_simple_audio_path(normalized_audio, generated_paths)

    normalized_image = _normalize_path(image_path)
    if not normalized_image:
        yield (
            gr.update(value=output_audio, visible=True),
            gr.update(value=None, visible=False),
            _final_status(status, output_audio, "Audio ready."),
            generated_files_update,
        )
        return

    audio_paths = audio_paths_for_video_export(output_audio, generated_paths)
    yield (
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=True),
        _video_export_status(audio_paths, video_resolution),
        generated_files_update,
    )

    artifacts: list[SimpleVideoArtifacts] = []
    try:
        for path in audio_paths:
            artifacts.append(
                export_simple_video_artifacts(
                    path,
                    normalized_image,
                    video_resolution,
                )
            )
    except (OSError, RuntimeError) as exc:
        logger.exception("Failed to create simple-tab MP4 preview")
        partial_files = _generated_files_with_video_artifacts(generated_paths, artifacts)
        yield (
            gr.update(value=output_audio, visible=True),
            gr.update(value=None, visible=False),
            f"{status or 'Generation complete.'}\nMP4 creation failed: {exc}",
            build_simple_generated_files_update(partial_files),
        )
        return

    final_files = _generated_files_with_video_artifacts(generated_paths, artifacts)
    yield (
        gr.update(value=None, visible=False),
        gr.update(value=artifacts[0].video_path, visible=True),
        _final_status(
            status,
            artifacts[0].video_path,
            _video_ready_message(artifacts),
            image_path=artifacts[0].image_path,
            media_count=len(artifacts),
        ),
        build_simple_generated_files_update(final_files),
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


def _flatten_paths(value: Any) -> list[str]:
    """Return normalized paths from nested Gradio file values."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        paths: list[str] = []
        for item in value:
            paths.extend(_flatten_paths(item))
        return paths
    path = _normalize_path(value)
    return [path] if path else []


def _generated_files_with_video_artifacts(
    generated_paths: list[str],
    artifacts: list[SimpleVideoArtifacts],
) -> list[str]:
    """Return generated file paths plus newly-created video artifacts."""

    video_paths = [artifact.video_path for artifact in artifacts]
    image_paths = [artifact.image_path for artifact in artifacts]
    return dedupe_paths([*generated_paths, *video_paths, *image_paths])


def _video_export_status(audio_paths: list[str], video_resolution: str | None) -> str:
    """Return a compact status while MP4 exports are running."""

    count = len(audio_paths)
    if count == 1:
        return (
            "Creating MP4 video from uploaded image...\n"
            f"Resolution: {video_resolution or '1080p'}"
        )
    return (
        f"Creating {count} MP4 videos from uploaded image...\n"
        f"Resolution: {video_resolution or '1080p'}"
    )


def _video_ready_message(artifacts: list[SimpleVideoArtifacts]) -> str:
    """Return the final ready message for one or more generated videos."""

    if len(artifacts) == 1:
        return "MP4 video ready."
    return f"{len(artifacts)} MP4 videos ready."


def _final_status(
    status: str | None,
    media_path: str,
    ready_message: str,
    *,
    image_path: str | None = None,
    media_count: int = 1,
) -> str:
    """Return compact final status with the saved run folder."""

    if status and "failed" in status.lower():
        return status.strip()

    run_folder = Path(media_path).expanduser().parent
    if image_path:
        media_label = "MP4" if media_count == 1 else "Preview MP4"
        return (
            f"{ready_message} Outputs are saved.\n"
            f"{media_label}: {media_path}\n"
            f"Image: {image_path}\n"
            f"Folder: {run_folder}"
        )
    return f"{ready_message}\nGeneration complete. Outputs are saved.\nFolder: {run_folder}"
