"""Shared cancellation state for training UI auto-label runs."""

from __future__ import annotations

import threading


_LOCK = threading.Lock()
_CANCEL_REQUESTED = False
_ACTIVE_INLINE_RUNS = 0


def clear_auto_label_cancel_request() -> None:
    """Clear any pending auto-label cancellation request."""

    global _CANCEL_REQUESTED  # noqa: PLW0603
    with _LOCK:
        _CANCEL_REQUESTED = False


def request_auto_label_cancel() -> None:
    """Mark the active auto-label run for cancellation."""

    global _CANCEL_REQUESTED  # noqa: PLW0603
    with _LOCK:
        _CANCEL_REQUESTED = True


def is_auto_label_cancel_requested() -> bool:
    """Return whether auto-label cancellation was requested."""

    with _LOCK:
        return _CANCEL_REQUESTED


def mark_inline_auto_label_started() -> None:
    """Record that an in-process auto-label run is active."""

    global _ACTIVE_INLINE_RUNS  # noqa: PLW0603
    with _LOCK:
        _ACTIVE_INLINE_RUNS += 1


def mark_inline_auto_label_finished() -> None:
    """Record that an in-process auto-label run has finished."""

    global _ACTIVE_INLINE_RUNS  # noqa: PLW0603
    with _LOCK:
        _ACTIVE_INLINE_RUNS = max(0, _ACTIVE_INLINE_RUNS - 1)


def has_active_inline_auto_label() -> bool:
    """Return whether an in-process auto-label run is active."""

    with _LOCK:
        return _ACTIVE_INLINE_RUNS > 0
