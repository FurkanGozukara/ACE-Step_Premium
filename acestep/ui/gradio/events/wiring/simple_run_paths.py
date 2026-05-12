"""Path resolution helpers for simple-tab generated media."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".opus", ".wav"}


def resolve_simple_audio_path(audio_path: str, generated_files: Any) -> str:
    """Return the original run-folder audio path when Gradio provides a cache path."""

    normalized_audio = _normalize_path(audio_path)
    if not normalized_audio:
        return ""

    generated_paths = _flatten_paths(generated_files)
    manifest_audio = _resolve_audio_from_manifest(normalized_audio, generated_paths)
    if manifest_audio:
        return manifest_audio

    listed_audio = _resolve_audio_from_file_list(normalized_audio, generated_paths)
    if listed_audio:
        return listed_audio

    return normalized_audio


def _resolve_audio_from_manifest(audio_path: str, generated_paths: list[str]) -> str:
    """Find the matching original audio path from generation manifests."""

    audio_name = Path(audio_path).name
    for manifest_path in generated_paths:
        if Path(manifest_path).name != "generation_manifest.json":
            continue
        manifest = _read_json(manifest_path)
        samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            candidate = _matching_sample_audio(sample, audio_path, audio_name)
            if candidate:
                return candidate
    return ""


def _matching_sample_audio(sample: dict[str, Any], audio_path: str, audio_name: str) -> str:
    """Return a sample audio path when it matches the Gradio cache path."""

    candidates = [sample.get("audio_path")]
    audio_paths = sample.get("audio_paths")
    if isinstance(audio_paths, dict):
        candidates.extend(audio_paths.values())

    normalized_candidates = [_normalize_path(candidate) for candidate in candidates]
    for candidate in normalized_candidates:
        if not candidate:
            continue
        if candidate == audio_path or Path(candidate).name == audio_name:
            return candidate
    return ""


def _resolve_audio_from_file_list(audio_path: str, generated_paths: list[str]) -> str:
    """Find the matching original audio path from generated file outputs."""

    audio_name = Path(audio_path).name
    audio_stem = Path(audio_path).stem
    for candidate in generated_paths:
        path = Path(candidate)
        if path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        if path.name == audio_name or path.stem == audio_stem:
            return _normalize_path(candidate)
    return ""


def _flatten_paths(value: Any) -> list[str]:
    """Return normalized path strings from nested Gradio file values."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        paths: list[str] = []
        for item in value:
            paths.extend(_flatten_paths(item))
        return paths
    path = _normalize_path(value)
    return [path] if path else []


def _normalize_path(value: Any) -> str:
    """Return a normalized path string from Gradio values."""

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
