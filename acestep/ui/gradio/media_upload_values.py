"""Helpers for normalizing Gradio upload component values."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def latest_upload_path(value: Any) -> str | None:
    """Return the newest file path from a Gradio upload value.

    Args:
        value: A Gradio upload value, including strings, FileData-like dicts,
            FileData-like objects, or stale single-file lists.

    Returns:
        The newest non-empty path, or ``None`` when no upload path is present.
    """

    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            path = latest_upload_path(item)
            if path:
                return path
        return None
    if isinstance(value, dict):
        for key in ("path", "name"):
            path = latest_upload_path(value.get(key))
            if path:
                return path
        return None
    for attr_name in ("path", "name"):
        path = latest_upload_path(getattr(value, attr_name, None))
        if path:
            return path
    return None


def resolve_effective_source_audio(
    source_value: Any,
    preview_value: Any = None,
    original_preview_value: Any = None,
) -> str | None:
    """Return the source-audio path, preferring a user-edited preview.

    Args:
        source_value: Original Source Audio upload value.
        preview_value: Current Source Audio Preview value.
        original_preview_value: Preview value created from the original upload.

    Returns:
        The edited preview path when the preview changed after upload, otherwise
        the original source upload path.
    """

    source_path = latest_upload_path(source_value)
    preview_path = latest_upload_path(preview_value)
    original_preview_path = latest_upload_path(original_preview_value)

    if preview_path:
        if original_preview_path:
            if _same_path(preview_path, original_preview_path):
                return source_path
            return preview_path
        if not source_path or not _same_path(preview_path, source_path):
            return preview_path
    return source_path


def _same_path(left: str, right: str) -> bool:
    """Return whether two upload paths refer to the same UI file value."""

    return left.replace("\\", "/").casefold() == right.replace("\\", "/").casefold()
