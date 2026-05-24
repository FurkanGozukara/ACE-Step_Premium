"""Shared cancellation state for training UI tensor preprocessing."""

from __future__ import annotations

import threading


_LOCK = threading.Lock()
_CANCEL_REQUESTED = False
_ACTIVE_INLINE_RUNS = 0


def clear_preprocess_cancel_request() -> None:
    """Clear any pending tensor-preprocess cancellation request."""

    global _CANCEL_REQUESTED  # noqa: PLW0603
    with _LOCK:
        _CANCEL_REQUESTED = False


def request_preprocess_cancel() -> None:
    """Mark the active tensor-preprocess run for cancellation."""

    global _CANCEL_REQUESTED  # noqa: PLW0603
    with _LOCK:
        _CANCEL_REQUESTED = True


def is_preprocess_cancel_requested() -> bool:
    """Return whether tensor-preprocess cancellation was requested."""

    with _LOCK:
        return _CANCEL_REQUESTED


def mark_inline_preprocess_started() -> None:
    """Record that an in-process tensor-preprocess run is active."""

    global _ACTIVE_INLINE_RUNS  # noqa: PLW0603
    with _LOCK:
        _ACTIVE_INLINE_RUNS += 1


def mark_inline_preprocess_finished() -> None:
    """Record that an in-process tensor-preprocess run has finished."""

    global _ACTIVE_INLINE_RUNS  # noqa: PLW0603
    with _LOCK:
        _ACTIVE_INLINE_RUNS = max(0, _ACTIVE_INLINE_RUNS - 1)


def has_active_inline_preprocess() -> bool:
    """Return whether an in-process tensor-preprocess run is active."""

    with _LOCK:
        return _ACTIVE_INLINE_RUNS > 0
