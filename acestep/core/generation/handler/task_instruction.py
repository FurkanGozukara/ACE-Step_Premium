"""Task instruction formatting helpers for generation UI and runtime."""

from __future__ import annotations

from typing import Sequence

from acestep.constants import TASK_INSTRUCTIONS


def generate_task_instruction(
    task_type: str,
    track_name: str | None = None,
    complete_track_classes: Sequence[str] | None = None,
) -> str:
    """Generate instruction text from task type and optional track context.

    Args:
        task_type: Backend task type such as ``text2music`` or ``repaint``.
        track_name: Optional single track name for extract/lego tasks.
        complete_track_classes: Optional track classes for complete tasks.

    Returns:
        The instruction string passed to the generation backend.
    """

    if task_type == "text2music":
        return TASK_INSTRUCTIONS["text2music"]
    if task_type == "repaint":
        return TASK_INSTRUCTIONS["repaint"]
    if task_type in ("cover", "cover-nofsq"):
        return TASK_INSTRUCTIONS["cover"]
    if task_type == "extract":
        if track_name:
            return TASK_INSTRUCTIONS["extract"].format(TRACK_NAME=track_name.upper())
        return TASK_INSTRUCTIONS["extract_default"]
    if task_type == "lego":
        if track_name:
            return TASK_INSTRUCTIONS["lego"].format(TRACK_NAME=track_name.upper())
        return TASK_INSTRUCTIONS["lego_default"]
    if task_type == "complete":
        if complete_track_classes:
            track_classes_upper = [track.upper() for track in complete_track_classes]
            return TASK_INSTRUCTIONS["complete"].format(
                TRACK_CLASSES=" | ".join(track_classes_upper)
            )
        return TASK_INSTRUCTIONS["complete_default"]
    return TASK_INSTRUCTIONS["text2music"]
