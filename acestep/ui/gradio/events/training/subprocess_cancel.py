"""UI cancellation handlers for isolated dataset worker subprocesses."""

from __future__ import annotations

from typing import Any

import gradio as gr
from loguru import logger

from .auto_label_control import (
    has_active_inline_auto_label,
    request_auto_label_cancel,
)
from .preprocess_control import (
    has_active_inline_preprocess,
    request_preprocess_cancel,
)
from .subprocess_control import request_training_subprocess_stop


AUTO_LABEL_CANCEL_CONFIRM_JS = (
    "() => { if (!confirm('Are you sure you want to cancel the current "
    "auto-label run?')) { throw new Error('Auto-label cancel aborted.'); } "
    "return []; }"
)
AUTO_LABEL_CANCEL_REQUESTED_STATUS = "Stopping isolated auto-label subprocess..."
NO_ACTIVE_AUTO_LABEL_STATUS = "No active isolated auto-label subprocess is currently running."
INLINE_AUTO_LABEL_CANCEL_REQUESTED_STATUS = (
    "Auto-label cancellation requested. In-process labeling will stop after the "
    "current file or batch step."
)
PREPROCESS_CANCEL_CONFIRM_JS = (
    "() => { if (!confirm('Are you sure you want to cancel the current tensor "
    "preprocess run?')) { throw new Error('Tensor preprocess cancel aborted.'); } "
    "return []; }"
)
PREPROCESS_CANCEL_REQUESTED_STATUS = "Stopping isolated tensor preprocess subprocess..."
NO_ACTIVE_PREPROCESS_STATUS = (
    "No active isolated tensor preprocess subprocess is currently running."
)
INLINE_PREPROCESS_CANCEL_REQUESTED_STATUS = (
    "Tensor preprocess cancellation requested. In-process preprocessing will stop "
    "after the current file or phase."
)


def request_auto_label_cancel_from_ui(confirmed: bool = True) -> str | Any:
    """Request cancellation for the active isolated auto-label worker."""

    if not confirmed:
        return gr.skip()

    request_auto_label_cancel()
    stopped_subprocess = request_training_subprocess_stop()
    if not stopped_subprocess:
        if has_active_inline_auto_label():
            logger.info("In-process auto-label cancellation requested from UI.")
            return INLINE_AUTO_LABEL_CANCEL_REQUESTED_STATUS
        logger.info("Auto-label cancel requested from UI, but no subprocess is active.")
        return NO_ACTIVE_AUTO_LABEL_STATUS

    logger.info("Auto-label subprocess cancellation requested from UI.")
    return AUTO_LABEL_CANCEL_REQUESTED_STATUS


def request_preprocess_cancel_from_ui(confirmed: bool = True) -> str | Any:
    """Request cancellation for the active tensor-preprocess worker."""

    if not confirmed:
        return gr.skip()

    request_preprocess_cancel()
    stopped_subprocess = request_training_subprocess_stop()
    if not stopped_subprocess:
        if has_active_inline_preprocess():
            logger.info("In-process tensor preprocess cancellation requested from UI.")
            return INLINE_PREPROCESS_CANCEL_REQUESTED_STATUS
        logger.info("Tensor preprocess cancel requested from UI, but no subprocess is active.")
        return NO_ACTIVE_PREPROCESS_STATUS

    logger.info("Tensor preprocess subprocess cancellation requested from UI.")
    return PREPROCESS_CANCEL_REQUESTED_STATUS
