"""Audio Processing-specific subprocess cancellation state."""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

from loguru import logger


class AudioProcessingCancelled(RuntimeError):
    """Raised when an Audio Processing subprocess is cancelled."""


class _AudioProcessingCancelState:
    """Track active Audio Processing subprocesses and cancellation requests."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._subprocesses: set[Any] = set()

    def request_cancel(self) -> bool:
        """Request cancellation and terminate registered subprocesses."""

        with self._lock:
            subprocesses = tuple(self._subprocesses)
            had_work = bool(subprocesses)
            if had_work:
                self._cancel_event.set()
        for process in subprocesses:
            terminate_audio_processing_process(process)
        return had_work

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._cancel_event.is_set()

    def register_subprocess(self, process: Any) -> None:
        """Register an Audio Processing subprocess for cancellation."""

        with self._lock:
            if not self._subprocesses:
                self._cancel_event.clear()
            self._subprocesses.add(process)

    def unregister_subprocess(self, process: Any) -> None:
        """Stop tracking an Audio Processing subprocess."""

        with self._lock:
            self._subprocesses.discard(process)
            if not self._subprocesses:
                self._cancel_event.clear()


_STATE = _AudioProcessingCancelState()


def request_audio_processing_cancel() -> bool:
    """Request cancellation for active Audio Processing subprocesses."""

    had_work = _STATE.request_cancel()
    if had_work:
        logger.info("[audio_processing_cancel] Cancellation requested.")
    return had_work


def is_audio_processing_cancelled() -> bool:
    """Return whether Audio Processing cancellation is active."""

    return _STATE.is_cancelled()


def register_audio_processing_subprocess(process: Any) -> None:
    """Register an Audio Processing subprocess for termination."""

    _STATE.register_subprocess(process)


def unregister_audio_processing_subprocess(process: Any) -> None:
    """Unregister an Audio Processing subprocess."""

    _STATE.unregister_subprocess(process)


def terminate_audio_processing_process(
    process: Any,
    *,
    timeout_seconds: float = 0.5,
) -> None:
    """Terminate an Audio Processing subprocess and kill it if it stalls."""

    if _poll_process(process) is not None:
        return
    try:
        logger.info("[audio_processing_cancel] Terminating isolated worker.")
        process.terminate()
    except Exception as exc:
        logger.warning("[audio_processing_cancel] Failed to terminate subprocess: {}", exc)
        return
    _wait_or_kill(process, timeout_seconds)


def _wait_or_kill(process: Any, timeout_seconds: float) -> None:
    """Wait briefly for process exit, then force-kill when needed."""

    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=timeout_seconds)
        return
    except TypeError:
        return
    except Exception:
        pass
    if _poll_process(process) is not None or _kill_windows_process_tree(process):
        return
    kill = getattr(process, "kill", None)
    if callable(kill):
        logger.info("[audio_processing_cancel] Killing isolated worker.")
        kill()


def _kill_windows_process_tree(process: Any) -> bool:
    """Force-kill a process tree on Windows when a PID is available."""

    if sys.platform != "win32":
        return False
    pid = getattr(process, "pid", None)
    if not pid:
        return False
    try:
        logger.info("[audio_processing_cancel] Killing isolated worker process tree.")
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return True
    except Exception as exc:
        logger.warning("[audio_processing_cancel] Failed to kill process tree: {}", exc)
        return False


def _poll_process(process: Any) -> Any:
    """Return process status, treating unsupported poll objects as active."""

    poll = getattr(process, "poll", None)
    if not callable(poll):
        return None
    try:
        return poll()
    except Exception:
        return None
