"""Filesystem helpers for Advanced-tab batch processing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from loguru import logger

from acestep.training.path_inputs import normalize_user_path
from acestep.ui.gradio.events.batch_folder_files import resolve_existing_input_folder
from acestep.ui.gradio.events.extract_stems import extract_stem_filename_suffix


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
        raise ValueError("Enter a Batch Process Output Folder before starting.")
    folder = Path(raw_value).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def discover_batch_extract_audio_files(
    input_folder: str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Find supported audio files in ``input_folder``."""

    folder = resolve_existing_input_folder(input_folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files = [
        path
        for path in sorted(iterator, key=lambda item: str(item.relative_to(folder)).lower())
        if path.is_file() and path.suffix.lower() in AUDIO_INPUT_SUFFIXES
    ]
    if not files:
        raise ValueError("No supported audio files found in the Batch Process input folder.")
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
    track_name: str | None = None,
    output_only: bool = False,
    overwrite_existing_files: bool = False,
) -> list[str]:
    """Copy generated audio files to ``output_folder`` using the source stem."""

    copied: list[str] = []
    copied_outputs: set[tuple[str, bool]] = set()
    for generated_path in generated_paths:
        source = Path(generated_path)
        if source.name == "generation_manifest.json":
            break
        suffix = source.suffix.lower()
        is_remaining = _is_remaining_audio(source)
        if output_only and is_remaining:
            continue
        output_key = (suffix, is_remaining)
        if (
            suffix not in AUDIO_OUTPUT_SUFFIXES
            or output_key in copied_outputs
            or not source.is_file()
        ):
            continue
        stem_suffix = extract_stem_filename_suffix(track_name)
        base_stem = f"{source_audio.stem}_{stem_suffix}" if stem_suffix else source_audio.stem
        target_stem = f"{base_stem}_remaining" if is_remaining else base_stem
        target = output_folder / f"{target_stem}{suffix}"
        if not overwrite_existing_files:
            target = _available_output_path(target)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(str(target))
        copied_outputs.add(output_key)
        if output_only and not is_remaining:
            break
    return copied


def _is_remaining_audio(path: Path) -> bool:
    """Return whether a generated audio path is an Extract remaining artifact."""

    return "_remaining" in path.stem.lower()


def _available_output_path(path: Path) -> Path:
    """Return a non-existing output path without overwriting user source files."""

    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_extract{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise ValueError(f"Could not find a free output filename near {path}.")
