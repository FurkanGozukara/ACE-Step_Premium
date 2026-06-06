"""Generated-song post-processing integration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from .file_processor import process_media_file
from .settings import AudioProcessingSettings


def postprocess_generated_sample(
    *,
    source_audio_path: str,
    run_dir: str | Path,
    key: str,
    settings: AudioProcessingSettings,
    original_audio_paths: dict[str, str],
) -> dict[str, Any]:
    """Post-process one generated audio file and return manifest metadata.

    Args:
        source_audio_path: Original generated audio used as the processing source.
        run_dir: Current generation run directory.
        key: Sample key used for output filenames.
        settings: Audio processing settings.
        original_audio_paths: Original generated files by format.

    Returns:
        JSON-safe metadata describing processed output and retained originals.
    """

    if not settings.enabled or not source_audio_path:
        return {"applied": False}

    try:
        result = process_media_file(
            source_audio_path,
            run_dir,
            settings,
            output_stem=f"{key}_postprocessed",
            include_video=False,
        )
    except Exception as exc:
        logger.exception("[audio_processing] Generated-song post-processing failed")
        return {"applied": False, "error": str(exc)}

    removed_originals = []
    if not settings.preserve_original:
        removed_originals = _remove_original_audio_paths(
            original_audio_paths,
            keep_path=result.audio_path,
        )

    return {
        "applied": True,
        "audio_path": result.audio_path,
        "metadata_path": result.metadata_path,
        "settings": settings.to_payload(),
        "original_audio_paths": dict(original_audio_paths),
        "removed_original_audio_paths": removed_originals,
        "lufs_before": result.processed_audio.lufs_before,
        "lufs_after": result.processed_audio.lufs_after,
        "duration_seconds": result.processed_audio.duration_seconds,
        "diffpitcher": result.processed_audio.diffpitcher_metadata,
        "trim": result.processed_audio.trim_metadata,
    }


def _remove_original_audio_paths(
    original_audio_paths: dict[str, str],
    *,
    keep_path: str,
) -> list[str]:
    """Remove original generated audio files when the user opts out of saving them."""

    keep = Path(keep_path).resolve()
    removed: list[str] = []
    for path_value in original_audio_paths.values():
        if not path_value:
            continue
        path = Path(path_value)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved == keep:
            continue
        try:
            if path.is_file():
                path.unlink()
                removed.append(str(path).replace("\\", "/"))
        except OSError as exc:
            logger.warning("[audio_processing] Could not remove original audio {}: {}", path, exc)
    return removed
