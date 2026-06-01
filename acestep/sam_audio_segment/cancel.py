"""SAM-Audio-specific cancellation state."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from loguru import logger

from acestep.core.generation.cancellation import CANCEL_MESSAGE, GenerationCancelled
from acestep.core.generation.subprocess_termination import terminate_generation_process


class _SamAudioCancelState:
    """Track active SAM-Audio work and kill registered subprocesses."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._active_count = 0
        self._subprocesses: set[Any] = set()

    def begin(self) -> None:
        """Enter an active SAM-Audio scope."""

        with self._lock:
            if self._active_count == 0 and not self._subprocesses:
                self._cancel_event.clear()
            self._active_count += 1

    def end(self) -> None:
        """Leave an active SAM-Audio scope."""

        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            if self._active_count == 0 and not self._subprocesses:
                self._cancel_event.clear()

    def request_cancel(self) -> bool:
        """Request cancellation and terminate every registered subprocess."""

        with self._lock:
            subprocesses = tuple(self._subprocesses)
            had_work = self._active_count > 0 or bool(subprocesses)
            if had_work:
                self._cancel_event.set()
        for process in subprocesses:
            terminate_generation_process(process)
        return had_work

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._cancel_event.is_set()

    def register_subprocess(self, process: Any) -> None:
        """Register a SAM-Audio subprocess for cancellation."""

        with self._lock:
            self._subprocesses.add(process)

    def unregister_subprocess(self, process: Any) -> None:
        """Stop tracking a SAM-Audio subprocess."""

        with self._lock:
            self._subprocesses.discard(process)
            if self._active_count == 0 and not self._subprocesses:
                self._cancel_event.clear()


_STATE = _SamAudioCancelState()


@contextmanager
def sam_audio_cancel_scope() -> Iterator[None]:
    """Scope a SAM-Audio foreground task for cancellation."""

    _STATE.begin()
    try:
        yield
    finally:
        _STATE.end()


def request_sam_audio_cancel() -> bool:
    """Request cancellation for active SAM-Audio work."""

    had_work = _STATE.request_cancel()
    if had_work:
        logger.info("[sam_audio_cancel] Cancellation requested.")
    return had_work


def check_sam_audio_cancelled() -> None:
    """Raise when a SAM-Audio cancellation request is active."""

    if _STATE.is_cancelled():
        raise GenerationCancelled(CANCEL_MESSAGE)


def is_sam_audio_cancelled() -> bool:
    """Return whether SAM-Audio cancellation is active."""

    return _STATE.is_cancelled()


def register_sam_audio_subprocess(process: Any) -> None:
    """Register a SAM-Audio subprocess for termination."""

    _STATE.register_subprocess(process)


def unregister_sam_audio_subprocess(process: Any) -> None:
    """Unregister a SAM-Audio subprocess."""

    _STATE.unregister_subprocess(process)
