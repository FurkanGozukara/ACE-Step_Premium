"""Filesystem helpers for Advanced-tab Batch Extract."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from loguru import logger

from acestep.training.path_inputs import normalize_user_path
from acestep.ui.gradio.events.batch_folder_files import resolve_existing_input_folder


AUDIO_INPUT_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
AUDIO_OUTPUT_SUFFIXES = AUDIO_INPUT_SUFFIXES


def resolve_batch_extract_output_folder(output_folder: str | Path) -> Path:
    """Return a writable output folder, requiring the user to provide it."""

    raw_value = normalize_user_path(output_folder)
    if not raw_value:
        raise ValueError("Enter a Batch Extract Output Folder before starting.")
    folder = Path(raw_value).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def discover_batch_extract_audio_files(input_folder: str | Path) -> list[Path]:
    """Find non-recursive supported audio files in ``input_folder``."""

    folder = resolve_existing_input_folder(input_folder)
    files = [
        path
        for path in sorted(folder.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in AUDIO_INPUT_SUFFIXES
    ]
    if not files:
        raise ValueError("No supported audio files found in the Batch Extract input folder.")
    return files


def audio_duration_seconds(path: Path) -> float:
    """Read an audio file duration for Extract metadata, falling back to auto duration."""

    try:
        import soundfile as sf

        duration = float(sf.info(str(path)).duration)
        return duration if duration > 0 else -1.0
    except Exception as exc:
        logger.warning("[batch_extract] Could not read duration for {}: {}", path, exc)
        return -1.0


def copy_batch_extract_audio_outputs(
    generated_paths: Sequence[str],
    source_audio: Path,
    output_folder: Path,
) -> list[str]:
    """Copy generated audio files to ``output_folder`` using the source stem."""

    copied: list[str] = []
    copied_suffixes: set[str] = set()
    for generated_path in generated_paths:
        source = Path(generated_path)
        if source.name == "generation_manifest.json":
            break
        suffix = source.suffix.lower()
        if suffix not in AUDIO_OUTPUT_SUFFIXES or suffix in copied_suffixes or not source.is_file():
            continue
        target = output_folder / f"{source_audio.stem}{suffix}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(str(target))
        copied_suffixes.add(suffix)
    return copied
