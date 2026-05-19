"""Persist per-sample labels for auto-labeled training datasets."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from acestep.training.path_safety import safe_path

from .models import AudioSample


def sample_label_metadata_path(audio_path: str) -> str:
    """Return the JSON sidecar path used to persist one audio sample label."""

    validated_audio_path = safe_path(audio_path)
    base_path = os.path.splitext(validated_audio_path)[0]
    return f"{base_path}.json"


def sample_label_metadata_output_path(
    sample: AudioSample,
    output_dir: str,
    source_root: str | None = None,
) -> str:
    """Return the processed-label JSON path inside a separate output folder.

    Args:
        sample: Labeled sample whose metadata will be saved.
        output_dir: Folder for processed label JSON files.
        source_root: Optional scanned dataset root. When provided, nested source
            folders are mirrored under ``output_dir`` to avoid filename clashes.

    Returns:
        Validated path for the sample label JSON inside ``output_dir``.
    """

    validated_output_dir = safe_path(output_dir)
    relative_label_path = _relative_label_path(sample, source_root)
    return safe_path(
        os.path.join(validated_output_dir, relative_label_path),
        base=validated_output_dir,
    )


def build_sample_label_metadata(sample: AudioSample) -> dict[str, Any]:
    """Build JSON metadata that ``scan_directory`` can load on the next run."""

    return {
        "audio_path": _metadata_audio_path(sample),
        "filename": sample.filename,
        "caption": sample.caption,
        "genre": sample.genre,
        "genres": sample.genre,
        "lyrics": sample.lyrics,
        "raw_lyrics": sample.raw_lyrics,
        "formatted_lyrics": sample.formatted_lyrics,
        "bpm": sample.bpm,
        "keyscale": sample.keyscale,
        "timesignature": sample.timesignature,
        "duration": sample.duration,
        "language": sample.language,
        "is_instrumental": sample.is_instrumental,
        "labeled": sample.labeled,
        "prompt_override": sample.prompt_override,
    }


def save_sample_label_metadata(
    sample: AudioSample,
    output_dir: str | None = None,
    source_root: str | None = None,
) -> str:
    """Write one sample label JSON atomically and return its path."""

    if output_dir:
        output_path = sample_label_metadata_output_path(sample, output_dir, source_root)
    else:
        output_path = sample_label_metadata_path(sample.audio_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_label_", suffix=".json", dir=output_dir or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(build_sample_label_metadata(sample), file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return output_path


def _relative_label_path(sample: AudioSample, source_root: str | None) -> str:
    """Return a collision-resistant relative JSON path for ``sample``."""

    if source_root:
        try:
            audio_path = safe_path(sample.audio_path)
            root_path = safe_path(source_root)
            if os.path.commonpath([root_path, audio_path]) == root_path:
                relative_audio = os.path.relpath(audio_path, root_path)
                return f"{os.path.splitext(relative_audio)[0]}.json"
        except (OSError, ValueError):
            pass

    stem = os.path.splitext(os.path.basename(sample.audio_path or sample.filename))[0]
    safe_stem = stem or sample.filename or "sample"
    unique_id = sample.id or "label"
    return f"{safe_stem}.{unique_id}.json"


def _metadata_audio_path(sample: AudioSample) -> str:
    """Return the canonical source audio path stored in processed-label metadata."""

    try:
        return safe_path(sample.audio_path)
    except (OSError, ValueError):
        return sample.audio_path
