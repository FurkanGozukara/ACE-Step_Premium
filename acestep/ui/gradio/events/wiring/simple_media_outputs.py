"""Output helpers for the simple Create tab media preview."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from loguru import logger

from ..results.output_manager import write_json
from ..results.video_export import copy_video_image_to_run_dir, create_still_image_video


def clear_simple_media_preview() -> tuple[Any, Any]:
    """Clear the simple-tab audio/video preview before a new generation."""

    return gr.update(value=None, visible=True), gr.update(value=None, visible=False)


def build_simple_media_preview(
    audio_path: Any,
    status: str | None,
    image_path: Any,
    video_resolution: str | None,
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

    normalized_image = _normalize_path(image_path)
    if not normalized_image:
        yield (
            gr.update(value=normalized_audio, visible=True),
            gr.update(value=None, visible=False),
            _final_status(status, normalized_audio, "Audio ready."),
        )
        return

    yield (
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=True),
        f"Creating MP4 video from uploaded image...\nResolution: {video_resolution or '1080p'}",
    )

    try:
        video_path = _create_video(normalized_audio, normalized_image, video_resolution)
    except RuntimeError as exc:
        logger.exception("Failed to create simple-tab MP4 preview")
        yield (
            gr.update(value=normalized_audio, visible=True),
            gr.update(value=None, visible=False),
            f"{status or 'Generation complete.'}\nMP4 creation failed: {exc}",
        )
        return

    yield (
        gr.update(value=None, visible=False),
        gr.update(value=video_path, visible=True),
        _final_status(status, video_path, "MP4 video ready."),
    )


def _create_video(audio_path: str, image_path: str, video_resolution: str | None) -> str:
    """Create an MP4 next to the generated audio and update metadata files."""

    audio = Path(audio_path).expanduser().resolve()
    run_dir = audio.parent
    image_copy = copy_video_image_to_run_dir(image_path, run_dir)
    output_path = run_dir / f"{audio.stem}_{str(video_resolution or '1080p').lower()}.mp4"
    video_path = create_still_image_video(
        image_path=image_copy,
        audio_path=str(audio),
        output_path=str(output_path),
        resolution=video_resolution or "1080p",
    )
    _update_generation_metadata(
        run_dir=run_dir,
        audio_path=str(audio).replace("\\", "/"),
        video_path=video_path,
        image_path=image_copy,
        video_resolution=video_resolution or "1080p",
    )
    return video_path


def _update_generation_metadata(
    *,
    run_dir: Path,
    audio_path: str,
    video_path: str,
    image_path: str,
    video_resolution: str,
) -> None:
    """Patch manifest and sidecar metadata with generated video paths."""

    manifest_path = run_dir / "generation_manifest.json"
    if not manifest_path.is_file():
        return

    manifest = _read_json(manifest_path)
    samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
    for sample in samples:
        if sample.get("audio_path") != audio_path:
            continue
        sample["video_path"] = video_path
        sample["video_image_path"] = image_path
        sample["video_resolution"] = video_resolution
        metadata_path = sample.get("metadata_path")
        if metadata_path:
            _update_sidecar(metadata_path, video_path, image_path, video_resolution)
        break

    request = manifest.setdefault("request", {})
    if isinstance(request, dict):
        request["video_image_path"] = image_path
        request["video_resolution"] = video_resolution
    write_json(manifest_path, manifest)


def _update_sidecar(
    metadata_path: str,
    video_path: str,
    image_path: str,
    video_resolution: str,
) -> None:
    """Patch a per-sample metadata sidecar with video export details."""

    target = Path(metadata_path)
    sidecar = _read_json(target)
    if not isinstance(sidecar, dict):
        return
    meta = sidecar.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta["video_path"] = video_path
        meta["video_image_path"] = image_path
        meta["video_resolution"] = video_resolution
    write_json(target, sidecar)


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty dict on failure."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_path(value: Any) -> str:
    """Return a normalized path string from Gradio file/image values."""

    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    return str(value or "").strip().replace("\\", "/")


def _final_status(status: str | None, media_path: str, ready_message: str) -> str:
    """Return compact final status with the saved run folder."""

    base = "Generation complete. Outputs are saved."
    if status and "failed" in status.lower():
        base = status.strip()
    run_folder = Path(media_path).expanduser().parent
    return f"{ready_message}\n{base}\nFolder: {run_folder}"
