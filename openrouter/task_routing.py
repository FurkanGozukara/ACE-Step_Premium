"""Task routing helpers for the legacy OpenRouter API server."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from acestep.core.generation.handler.task_instruction import generate_task_instruction

NO_LM_TASK_TYPES = frozenset({"cover", "cover-nofsq", "repaint", "extract"})


def _normalize_track_classes(track_classes: Any) -> list[str] | None:
    """Normalize OpenRouter track class input into a list of non-empty strings."""

    if track_classes is None:
        return None
    if isinstance(track_classes, str):
        values: Sequence[Any] = [track_classes]
    elif isinstance(track_classes, Sequence):
        values = track_classes
    else:
        values = [track_classes]
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return normalized or None


def resolve_task_instruction(
    task_type: str,
    track_name: str | None = None,
    track_classes: Any = None,
) -> str:
    """Return placeholder-safe DiT instruction text for an OpenRouter task."""

    clean_track_name = str(track_name or "").strip() or None
    return generate_task_instruction(
        task_type,
        track_name=clean_track_name,
        complete_track_classes=_normalize_track_classes(track_classes),
    )


def task_skips_lm(task_type: str) -> bool:
    """Return whether a task should bypass LM/CoT prompting in OpenRouter."""

    return str(task_type or "").strip() in NO_LM_TASK_TYPES
