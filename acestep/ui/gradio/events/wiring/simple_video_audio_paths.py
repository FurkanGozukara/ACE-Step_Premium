"""Audio path selection for simple-tab multi-song video exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".opus", ".wav"}
_AUDIO_SUFFIX_PRIORITY = {
    ".flac": 0,
    ".wav": 1,
    ".mp3": 2,
    ".m4a": 3,
    ".aac": 4,
    ".opus": 5,
}


def audio_paths_for_video_export(output_audio: str, generated_paths: list[str]) -> list[str]:
    """Return the primary generated audio path for each song in output order."""

    manifest_paths = _audio_paths_from_manifests(generated_paths)
    listed_paths = _audio_paths_from_file_list(generated_paths)
    audio_paths = manifest_paths or listed_paths
    return dedupe_paths([output_audio, *audio_paths])


def dedupe_paths(paths: list[str]) -> list[str]:
    """Return path strings without duplicates while preserving order."""

    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = _normalize_path(path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)
    return unique_paths


def _audio_paths_from_manifests(generated_paths: list[str]) -> list[str]:
    """Read generated primary audio paths from run manifest files."""

    audio_paths: list[str] = []
    for manifest_path in generated_paths:
        if Path(manifest_path).name != "generation_manifest.json":
            continue
        manifest = _read_json(manifest_path)
        samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if isinstance(sample, dict):
                candidate = _sample_primary_audio_path(sample)
                if candidate:
                    audio_paths.append(candidate)
    return dedupe_paths(audio_paths)


def _sample_primary_audio_path(sample: dict[str, Any]) -> str:
    """Return the best primary audio path recorded for one manifest sample."""

    primary_path = _normalize_path(sample.get("audio_path"))
    if primary_path:
        return primary_path

    audio_paths = sample.get("audio_paths")
    if not isinstance(audio_paths, dict):
        return ""
    for suffix in _AUDIO_SUFFIX_PRIORITY:
        for value in audio_paths.values():
            path = _normalize_path(value)
            if path and Path(path).suffix.lower() == suffix:
                return path
    return ""


def _audio_paths_from_file_list(generated_paths: list[str]) -> list[str]:
    """Return one preferred audio path per generated song from file outputs."""

    by_stem: dict[str, list[str]] = {}
    for candidate in generated_paths:
        path = Path(candidate)
        if path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        by_stem.setdefault(path.stem, []).append(_normalize_path(candidate))

    audio_paths: list[str] = []
    for candidates in by_stem.values():
        audio_paths.append(min(candidates, key=_audio_suffix_rank))
    return dedupe_paths(audio_paths)


def _audio_suffix_rank(path: str) -> int:
    """Return the preferred format order for a generated audio path."""

    return _AUDIO_SUFFIX_PRIORITY.get(
        Path(path).suffix.lower(),
        len(_AUDIO_SUFFIX_PRIORITY),
    )


def _normalize_path(value: Any) -> str:
    """Return a normalized path string from Gradio-like values."""

    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    elif hasattr(value, "path"):
        value = getattr(value, "path")
    elif hasattr(value, "name"):
        value = getattr(value, "name")
    return str(value or "").strip().replace("\\", "/")


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty dict on failure."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}
