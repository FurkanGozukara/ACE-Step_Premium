"""Artifact persistence for simple-tab still-image MP4 exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..results.output_manager import write_json
from ..results.video_export import copy_video_image_to_run_dir, create_still_image_video


@dataclass(frozen=True)
class SimpleVideoArtifacts:
    """Paths written for a simple-tab video export."""

    video_path: str
    image_path: str


def export_simple_video_artifacts(
    audio_path: str,
    image_path: str,
    video_resolution: str | None,
) -> SimpleVideoArtifacts:
    """Create and record the MP4 plus the uploaded image in the audio run folder."""

    audio = Path(audio_path).expanduser().resolve()
    run_dir = audio.parent
    resolution = str(video_resolution or "1080p")
    image_copy = copy_video_image_to_run_dir(image_path, run_dir)
    output_path = run_dir / f"{audio.stem}_{resolution.lower()}.mp4"
    video_path = create_still_image_video(
        image_path=image_copy,
        audio_path=str(audio),
        output_path=str(output_path),
        resolution=resolution,
    )
    _update_generation_metadata(
        run_dir=run_dir,
        audio_path=str(audio).replace("\\", "/"),
        video_path=video_path,
        image_path=image_copy,
        video_resolution=resolution,
    )
    logger.info("Saved simple-tab MP4 video: {}", video_path)
    logger.info("Saved simple-tab video image: {}", image_copy)
    return SimpleVideoArtifacts(video_path=video_path, image_path=image_copy)


def _update_generation_metadata(
    *,
    run_dir: Path,
    audio_path: str,
    video_path: str,
    image_path: str,
    video_resolution: str,
) -> None:
    """Patch run-level and per-sample metadata with generated video paths."""

    _update_generation_manifest(
        run_dir=run_dir,
        audio_path=audio_path,
        video_path=video_path,
        image_path=image_path,
        video_resolution=video_resolution,
    )
    _update_generation_request(
        run_dir=run_dir,
        video_path=video_path,
        image_path=image_path,
        video_resolution=video_resolution,
    )


def _update_generation_manifest(
    *,
    run_dir: Path,
    audio_path: str,
    video_path: str,
    image_path: str,
    video_resolution: str,
) -> None:
    """Patch the completed generation manifest when it exists."""

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
        request["video_path"] = video_path
        request["video_image_path"] = image_path
        request["video_resolution"] = video_resolution
    write_json(manifest_path, manifest)


def _update_generation_request(
    *,
    run_dir: Path,
    video_path: str,
    image_path: str,
    video_resolution: str,
) -> None:
    """Patch the persisted request snapshot with simple video export details."""

    request_path = run_dir / "generation_request.json"
    if not request_path.is_file():
        return

    payload = _read_json(request_path)
    request = payload.setdefault("request", {})
    assets = payload.setdefault("assets", {})
    if isinstance(request, dict):
        request["video_path"] = video_path
        request["video_image_path"] = image_path
        request["video_resolution"] = video_resolution
    if isinstance(assets, dict):
        assets["video_path"] = video_path
        assets["video_image_path"] = image_path
    write_json(request_path, payload)


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
