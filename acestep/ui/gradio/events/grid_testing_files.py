"""Output-folder and artifact flattening helpers for Grid Testing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from acestep.ui.gradio.events.results.output_manager import (
    make_json_safe,
    write_json,
    write_text,
)
from acestep.ui.gradio.events.grid_testing_paths import next_sample_index


_AUDIO_SUFFIXES = {".aac", ".flac", ".mp3", ".opus", ".wav"}


def flatten_generation_outputs(
    generated_paths: list[str],
    target_folder: Path,
    *,
    prefix: str,
    caption: Any,
    lyrics: Any,
    mp3_only: bool,
) -> list[str]:
    """Copy one temporary generation run into the flat grid output layout.

    Args:
        generated_paths: Artifact paths returned by the generation runner.
        target_folder: Final grid output folder.
        prefix: Filename prefix for this LoRA or base-model job.
        caption: Current style prompt saved beside metadata outputs.
        lyrics: Current lyrics saved beside metadata outputs.
        mp3_only: Whether only MP3 audio files should be kept.

    Returns:
        Final artifact paths written directly under ``target_folder``.

    Raises:
        ValueError: If no manifest samples or required MP3 files are available.
    """

    target_folder.mkdir(parents=True, exist_ok=True)
    samples = _manifest_samples(generated_paths)
    if not samples:
        samples = _fallback_samples(generated_paths)
    if not samples:
        raise ValueError("Generation completed without discoverable output files.")

    written: list[str] = []
    next_index = next_sample_index(target_folder, prefix)
    for offset, sample in enumerate(samples):
        final_stem = f"{prefix}-{next_index + offset:04d}"
        if mp3_only:
            mp3_path = _sample_mp3_path(sample)
            if not mp3_path:
                raise ValueError("MP3-only grid output requested, but no MP3 was generated.")
            written.append(_copy_file(mp3_path, target_folder / f"{final_stem}.mp3"))
            continue

        audio_paths = _copy_sample_audio(sample, target_folder, final_stem)
        written.extend(audio_paths.values())
        metadata_path = _copy_sample_metadata(sample, target_folder, final_stem, audio_paths)
        if metadata_path:
            written.append(metadata_path)
        written.append(write_text(target_folder / f"{final_stem}_caption.txt", caption))
        written.append(write_text(target_folder / f"{final_stem}_lyrics.txt", lyrics))

    return written


def write_grid_manifest(
    target_folder: Path,
    rows: list[dict[str, Any]],
) -> str:
    """Write the top-level Grid Testing manifest."""

    return write_json(
        target_folder / "grid_manifest.json",
        {
            "_meta": {
                "format": "ace_step_grid_testing_manifest",
                "version": 1,
            },
            "items": make_json_safe(rows),
        },
    )


def _manifest_samples(generated_paths: list[str]) -> list[dict[str, Any]]:
    """Return sample rows from the generated run manifest when present."""

    for raw_path in generated_paths:
        path = Path(str(raw_path))
        if path.name != "generation_manifest.json" or not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        samples = payload.get("samples", [])
        return samples if isinstance(samples, list) else []
    return []


def _fallback_samples(generated_paths: list[str]) -> list[dict[str, Any]]:
    """Build minimal sample rows when a manifest is unavailable."""

    samples = []
    for raw_path in generated_paths:
        path = Path(str(raw_path))
        if path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        samples.append({"audio_paths": {path.suffix.lower().lstrip("."): str(path)}})
    return samples


def _sample_mp3_path(sample: dict[str, Any]) -> str:
    """Return the MP3 path for a sample row."""

    audio_paths = sample.get("audio_paths") if isinstance(sample, dict) else {}
    if isinstance(audio_paths, dict):
        return str(audio_paths.get("mp3") or sample.get("mp3_path") or "").strip()
    return str(sample.get("mp3_path") or "").strip()


def _copy_sample_audio(
    sample: dict[str, Any],
    target_folder: Path,
    final_stem: str,
) -> dict[str, str]:
    """Copy all audio formats for one sample and return final paths by format."""

    copied: dict[str, str] = {}
    audio_paths = sample.get("audio_paths") if isinstance(sample, dict) else {}
    if not isinstance(audio_paths, dict):
        audio_paths = {}
    for audio_format, raw_path in audio_paths.items():
        source = Path(str(raw_path))
        if not source.is_file():
            continue
        target = target_folder / f"{final_stem}{source.suffix.lower()}"
        copied[str(audio_format)] = _copy_file(source, target)
    return copied


def _copy_sample_metadata(
    sample: dict[str, Any],
    target_folder: Path,
    final_stem: str,
    audio_paths: dict[str, str],
) -> str | None:
    """Copy and patch one sample JSON sidecar for the flattened grid layout."""

    source = Path(str(sample.get("metadata_path") or ""))
    if not source.is_file():
        return None
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.setdefault("_meta", {})
    primary_audio = next(iter(audio_paths.values()), "")
    payload["audio_paths"] = audio_paths
    for audio_format, audio_path in audio_paths.items():
        payload[f"{audio_format}_path"] = audio_path
    meta["audio_path"] = primary_audio
    meta["audio_paths"] = audio_paths
    meta["grid_output_stem"] = final_stem
    meta["run_dir"] = str(target_folder.resolve()).replace("\\", "/")
    return write_json(target_folder / f"{final_stem}.json", payload)


def _copy_file(source_path: str | Path, target_path: str | Path) -> str:
    """Copy a file to the target path and return its normalized absolute path."""

    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target.resolve()).replace("\\", "/")
