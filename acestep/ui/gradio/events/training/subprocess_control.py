"""Track and stop active isolated training subprocesses."""

from __future__ import annotations

import subprocess
import threading

from loguru import logger

from acestep.core.generation.subprocess_termination import terminate_generation_process


_LOCK = threading.Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_STOP_REQUESTED = False
_STOP_TIMEOUT_SECONDS = 0.5


def register_training_subprocess(process: subprocess.Popen) -> None:
    """Register a training worker process as active."""

    global _STOP_REQUESTED  # noqa: PLW0603
    with _LOCK:
        _STOP_REQUESTED = False
        _ACTIVE_PROCESSES.add(process)


def unregister_training_subprocess(process: subprocess.Popen) -> None:
    """Remove a training worker process from the active registry."""

    with _LOCK:
        _ACTIVE_PROCESSES.discard(process)


def request_training_subprocess_stop() -> bool:
    """Terminate active training subprocesses and return whether any existed."""

    global _STOP_REQUESTED  # noqa: PLW0603
    with _LOCK:
        processes = list(_ACTIVE_PROCESSES)
        if not processes:
            return False
        _STOP_REQUESTED = True

    for process in processes:
        if process.poll() is not None:
            unregister_training_subprocess(process)
            continue
        logger.info("Stopping isolated training subprocess pid={}", process.pid)
        terminate_generation_process(process, timeout_seconds=_STOP_TIMEOUT_SECONDS)
    return True


def consume_training_subprocess_stop_requested() -> bool:
    """Return and clear whether a subprocess stop was requested."""

    global _STOP_REQUESTED  # noqa: PLW0603
    with _LOCK:
        requested = _STOP_REQUESTED
        _STOP_REQUESTED = False
    return requested
