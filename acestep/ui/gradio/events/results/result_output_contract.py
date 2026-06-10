"""Shared output ordering for generation result tuples."""

from pathlib import Path
from typing import Any


AUDIO_SLOT_COUNT = 8
CORE_OUTPUT_COUNT = 54
ALL_AUDIO_PATHS_INDEX = 16
GENERATION_INFO_INDEX = 17
STATUS_INDEX = 18
SEED_INDEX = 19
SCORES_START_INDEX = 20
CODES_START_INDEX = 28
DETAILS_START_INDEX = 36
LRC_START_INDEX = 44
LM_METADATA_INDEX = 52
IS_FORMAT_CAPTION_INDEX = 53
EXTRA_OUTPUTS_INDEX = 54
CODES_LIST_INDEX = 55

SOURCE_COMPARE_TASKS = frozenset({"cover", "cover-nofsq", "repaint", "lego", "complete"})
BOUNDED_SOURCE_EDIT_TASKS = frozenset({"repaint", "lego", "complete"})


def should_show_source_audio(task_type: str | None) -> bool:
    """Return whether a task should display original source audio beside output."""
    return str(task_type or "").strip().lower() in SOURCE_COMPARE_TASKS


def is_bounded_source_edit(
    task_type: str | None,
    repainting_start: float | int | str | None,
    repainting_end: float | int | str | None,
) -> bool:
    """Return whether a source-edit task targets a bounded source-audio range."""
    if str(task_type or "").strip().lower() not in BOUNDED_SOURCE_EDIT_TASKS:
        return False
    try:
        start = float(repainting_start)
        end = float(repainting_end)
    except (TypeError, ValueError):
        return False
    return end > start


def source_audio_update_path(
    task_type: str | None,
    source_audio_path: str | None,
) -> str | None:
    """Return normalized source path for result comparison players."""
    if not should_show_source_audio(task_type):
        return None
    path = str(source_audio_path or "").strip()
    return path.replace("\\", "/") if path else None


def source_audio_paths_for_slots(
    task_type: str | None,
    source_audio_path: str | None,
    slot_count: int = AUDIO_SLOT_COUNT,
) -> list[str | None]:
    """Return per-slot source paths for visible comparison players."""
    path = source_audio_update_path(task_type, source_audio_path)
    return [path] * slot_count if path else [None] * slot_count


def extract_source_audio_path(all_audio_paths: Any) -> str | None:
    """Find the copied source-audio asset in a generation file list."""
    if not isinstance(all_audio_paths, (list, tuple)):
        return None
    for path in all_audio_paths:
        path_text = str(path or "")
        if not path_text:
            continue
        stem = Path(path_text).stem.lower()
        if stem == "source_audio":
            return path_text.replace("\\", "/")
    return None


def extract_latest_edit_area_paths(all_audio_paths: Any) -> tuple[str | None, str | None]:
    """Find latest edited-area generated/original clip paths in a file list."""

    if not isinstance(all_audio_paths, (list, tuple)):
        return None, None

    generated_path = None
    original_path = None
    for path in all_audio_paths:
        path_text = str(path or "")
        if not path_text:
            continue
        stem = Path(path_text).stem.lower()
        normalized = path_text.replace("\\", "/")
        if stem.endswith("_latest_repainted_area_original"):
            original_path = normalized
        elif stem.endswith("_latest_repainted_area"):
            generated_path = normalized
    return generated_path, original_path
