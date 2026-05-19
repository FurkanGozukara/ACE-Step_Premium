"""Persist per-sample labels beside source audio files."""

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


def build_sample_label_metadata(sample: AudioSample) -> dict[str, Any]:
    """Build JSON metadata that ``scan_directory`` can load on the next run."""

    return {
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


def save_sample_label_metadata(sample: AudioSample) -> str:
    """Write one sample label JSON atomically and return the sidecar path."""

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
