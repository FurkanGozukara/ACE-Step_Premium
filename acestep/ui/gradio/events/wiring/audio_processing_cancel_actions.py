"""Audio Processing UI cancellation helpers."""

from __future__ import annotations

from typing import Any

import gradio as gr
from loguru import logger

from acestep.audio_processing.cancel import request_audio_processing_cancel


AUDIO_PROCESSING_CANCEL_CONFIRM_JS = (
    "() => [confirm('Are you sure you want to cancel the current Audio Processing run?')]"
)
AUDIO_PROCESSING_CANCEL_REQUESTED_STATUS = (
    "Audio Processing subprocess cancellation requested. The isolated worker is being stopped."
)
AUDIO_PROCESSING_NO_ACTIVE_STATUS = (
    "No active Audio Processing subprocess is currently running."
)
AUDIO_PROCESSING_IN_PROCESS_STATUS = (
    "Subprocess mode is off. The current in-process Audio Processing run cannot be "
    "interrupted safely."
)


def request_audio_processing_cancel_from_ui(
    confirmed: bool,
    subprocess_mode_enabled: bool = False,
) -> str | Any:
    """Request Audio Processing subprocess cancellation after browser confirmation."""

    if not confirmed:
        return gr.skip()
    if not subprocess_mode_enabled:
        return AUDIO_PROCESSING_IN_PROCESS_STATUS
    had_active_work = request_audio_processing_cancel()
    if not had_active_work:
        logger.info(
            "[audio_processing_cancel] Cancel requested, but no subprocess is active."
        )
        return AUDIO_PROCESSING_NO_ACTIVE_STATUS
    logger.info("[audio_processing_cancel] Cancellation requested from UI.")
    return AUDIO_PROCESSING_CANCEL_REQUESTED_STATUS
