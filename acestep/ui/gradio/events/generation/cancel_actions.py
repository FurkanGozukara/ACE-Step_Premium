"""UI handlers for user-requested generation cancellation."""

from __future__ import annotations

from typing import Any

import gradio as gr
from loguru import logger

from acestep.core.generation.cancellation import request_generation_cancel


CANCEL_CONFIRM_JS = "() => [confirm('Are you sure you want to cancel the current generation?')]"
BATCH_CANCEL_CONFIRM_JS = (
    "() => [confirm('Are you sure you want to cancel the current generation "
    "and the remaining batch?')]"
)
CANCEL_REQUESTED_STATUS = (
    "Subprocess cancellation requested. The isolated worker is being stopped."
)
NO_ACTIVE_GENERATION_STATUS = "No active subprocess generation is currently running."
SUBPROCESS_MODE_DISABLED_STATUS = "Subprocess mode is off. Nothing was cancelled."


def request_generation_cancel_from_ui(
    confirmed: bool,
    subprocess_mode_enabled: bool = False,
) -> str | Any:
    """Request generation cancellation after browser confirmation.

    Args:
        confirmed: Browser confirmation result from the cancel button.

    Returns:
        Status text for the calling screen, or ``gr.skip()`` when cancelled.
    """

    if not confirmed:
        return gr.skip()
    if not subprocess_mode_enabled:
        return SUBPROCESS_MODE_DISABLED_STATUS
    had_active_work = request_generation_cancel(subprocess_only=True)
    if not had_active_work:
        logger.info("[generation_cancel] Cancel requested from UI, but no subprocess is active.")
        return NO_ACTIVE_GENERATION_STATUS
    logger.info("[generation_cancel] Cancellation requested from UI.")
    return CANCEL_REQUESTED_STATUS


def request_generation_cancel_pair_from_ui(
    confirmed: bool,
    subprocess_mode_enabled: bool = False,
) -> tuple[Any, Any]:
    """Return the cancellation status for two UI status components."""

    status = request_generation_cancel_from_ui(confirmed, subprocess_mode_enabled)
    return status, status
