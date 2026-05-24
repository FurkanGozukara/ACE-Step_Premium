"""Hydrate scanned samples from processed auto-label JSON files."""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

from acestep.training.path_safety import safe_path

from .models import AudioSample


def hydrate_samples_from_label_dir(samples: list[AudioSample], label_dir: str) -> int:
    """Apply matching processed labels to scanned samples.

    Args:
        samples: Scanned samples to update in place.
        label_dir: Folder containing processed auto-label JSON files.

    Returns:
        Number of samples hydrated from existing labels.
    """

    try:
        validated_dir = safe_path(label_dir)
    except ValueError:
        return 0
    if not os.path.isdir(validated_dir):
        return 0

    labels_by_audio_path = _load_labels_by_audio_path(validated_dir)
    hydrated_count = 0
    for sample in samples:
        label = labels_by_audio_path.get(_path_key(sample.audio_path))
        if label is None or not _is_usable_label(label):
            continue
        _apply_label(sample, label)
        hydrated_count += 1
    return hydrated_count


def has_unlabeled_samples(samples: list[AudioSample]) -> bool:
    """Return whether any sample still needs auto-label work."""

    return any(_needs_label(sample) for sample in samples)


def _load_labels_by_audio_path(label_dir: str) -> dict[str, dict[str, Any]]:
    """Return processed-label metadata keyed by canonical audio path."""

    labels: dict[str, dict[str, Any]] = {}
    for root, _, files in os.walk(label_dir):
        for filename in files:
            if not filename.lower().endswith(".json"):
                continue
            label_path = os.path.join(root, filename)
            label = _read_label(label_path)
            if label is None:
                continue
            key = _path_key(str(label.get("audio_path") or ""))
            if key:
                labels[key] = label
    return labels


def _needs_label(sample: AudioSample) -> bool:
    """Return whether a sample has no completed label."""

    caption = getattr(sample, "caption", "")
    return not getattr(sample, "labeled", False) or not bool(caption and caption.strip())


def _read_label(label_path: str) -> dict[str, Any] | None:
    """Read one processed-label JSON file."""

    try:
        with open(label_path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except Exception as exc:
        logger.warning(f"Failed to read processed label {label_path}: {exc}")
        return None
    if not isinstance(data, dict) or not data.get("audio_path"):
        return None
    return data


def _apply_label(sample: AudioSample, label: dict[str, Any]) -> None:
    """Apply processed-label fields to a scanned sample."""

    string_fields = (
        "caption",
        "lyrics",
        "raw_lyrics",
        "formatted_lyrics",
        "keyscale",
        "timesignature",
        "caption_source",
    )
    for field_name in string_fields:
        if field_name in label:
            setattr(sample, field_name, str(label.get(field_name) or ""))

    genre = label.get("genre") or label.get("genres")
    if genre is not None:
        sample.genre = str(genre)

    language = label.get("language") or label.get("vocal_language")
    if language is not None:
        sample.language = str(language)

    if "bpm" in label:
        sample.bpm = label["bpm"]
    if "duration" in label and label["duration"]:
        sample.duration = label["duration"]
    if "prompt_override" in label:
        sample.prompt_override = label["prompt_override"]
    if "is_instrumental" in label:
        sample.is_instrumental = bool(label["is_instrumental"])
    elif "instrumental" in label:
        sample.is_instrumental = bool(label["instrumental"])

    sample.labeled = bool(label.get("labeled")) or bool(sample.caption.strip())


def _is_usable_label(label: dict[str, Any]) -> bool:
    """Return whether a processed label has enough data to skip auto-labeling."""

    return bool(str(label.get("caption") or "").strip())


def _path_key(path: str) -> str:
    """Return a normalized path key for matching label metadata to samples."""

    if not path:
        return ""
    try:
        validated_path = safe_path(path)
    except ValueError:
        return ""
    return os.path.normcase(os.path.normpath(validated_path))
