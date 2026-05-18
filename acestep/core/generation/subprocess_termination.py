"""Process termination helpers for isolated generation workers."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from loguru import logger


def terminate_generation_process(
    process: Any,
    *,
    timeout_seconds: float = 0.5,
) -> None:
    """Terminate a generation subprocess and kill it if it does not exit quickly.

    Args:
        process: ``subprocess.Popen``-like object to stop.
        timeout_seconds: Seconds to wait after graceful termination before kill.
    """

    if _poll_process(process) is not None:
        return

    try:
        logger.info("[generation_cancel] Terminating isolated generation worker.")
        process.terminate()
    except Exception as exc:  # pragma: no cover - defensive subprocess cleanup.
        logger.warning("[generation_cancel] Failed to terminate subprocess: {}", exc)
        return

    _wait_or_kill(process, timeout_seconds)


def _wait_or_kill(process: Any, timeout_seconds: float) -> None:
    """Wait briefly for process exit, then force-kill when supported."""

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

    if _poll_process(process) is not None:
        return

    if _kill_windows_process_tree(process):
        return

    kill = getattr(process, "kill", None)
    if not callable(kill):
        return
    try:
        logger.info("[generation_cancel] Killing isolated generation worker.")
        kill()
    except Exception as exc:  # pragma: no cover - defensive subprocess cleanup.
        logger.warning("[generation_cancel] Failed to kill subprocess: {}", exc)


def _kill_windows_process_tree(process: Any) -> bool:
    """Force-kill a process tree on Windows when a PID is available."""

    if sys.platform != "win32":
        return False
    pid = getattr(process, "pid", None)
    if not pid:
        return False
    try:
        logger.info("[generation_cancel] Killing isolated generation process tree.")
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive subprocess cleanup.
        logger.warning("[generation_cancel] Failed to kill process tree: {}", exc)
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
