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
