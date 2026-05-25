"""Preview serialization helpers for imported dataset samples."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

from acestep.training.dataset_builder_modules.models import AudioSample


EMPTY_DATASET_PREVIEW = ("", "{}", None, None, None)


def sample_title(sample: AudioSample) -> str:
    """Return a readable song title for a dataset sample."""

    source = sample.filename or _path_name(sample.audio_path) or sample.id or "Untitled song"
    return Path(source).stem if Path(source).suffix else source


def preview_sample(sample: AudioSample, index: int) -> tuple[str, str, str, None, None]:
    """Return summary text, JSON metadata, and audio values for one sample."""

    payload: dict[str, Any] = sample.to_dict()
    payload["index"] = index
    instruction = sample.caption or sample.genre or sample_title(sample)
    return (
        instruction,
        json.dumps(payload, indent=2, ensure_ascii=False),
        sample.audio_path,
        None,
        None,
    )


def _path_name(path: str) -> str:
    """Return a basename for native or Windows-style paths."""

    if not path:
        return ""
    name = Path(path).name
    if name == path:
        name = PureWindowsPath(path).name
    return name
