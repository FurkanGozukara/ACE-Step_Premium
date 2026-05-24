"""Progress helpers for batched dataset auto-labeling."""

from __future__ import annotations

from typing import Callable

from .label_progress import LabelProgressTracker


def start_batch_item(
    tracker: LabelProgressTracker,
    progress_callback: Callable[[str], None] | None,
    *,
    position: int,
    labeled_count: int,
    left_count: int,
    filename: str,
) -> str:
    """Emit the start progress line for a sample."""

    tracker.begin_item()
    message = tracker.start_message(position, labeled_count, left_count, filename)
    if progress_callback:
        progress_callback(message)
    return message


def complete_batch_item(
    tracker: LabelProgressTracker,
    progress_callback: Callable[[str], None] | None,
    *,
    position: int,
    labeled_count: int,
    left_count: int,
    filename: str,
) -> None:
    """Emit the completion progress line for a sample."""

    tracker.complete_item()
    if progress_callback:
        progress_callback(
            tracker.complete_message(position, labeled_count, left_count, filename)
        )


def normalize_auto_label_batch_size(value: object) -> int:
    """Clamp a user-provided auto-label batch size to the supported range."""

    try:
        return max(1, min(99, int(value or 1)))
    except (TypeError, ValueError):
        return 1
