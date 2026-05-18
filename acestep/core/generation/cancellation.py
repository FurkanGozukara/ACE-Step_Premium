"""Cooperative cancellation state for foreground music generation."""

from __future__ import annotations

import gc
import sys
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from loguru import logger

from acestep.core.generation.subprocess_termination import terminate_generation_process


CANCEL_MESSAGE = "Generation cancelled by user."


class GenerationCancelled(RuntimeError):
    """Raised when a user-requested generation cancellation is observed."""


class _GenerationCancellationState:
    """Thread-safe cancellation registry shared by Gradio generation paths."""

    def __init__(self) -> None:
        """Initialize cancellation flags, active scope count, and subprocess set."""

        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._active_count = 0
        self._subprocesses: set[Any] = set()

    def begin(self) -> None:
        """Enter a generation scope, clearing stale cancellation when outermost."""

        with self._lock:
            if self._active_count == 0:
                self._cancel_event.clear()
            self._active_count += 1

    def end(self) -> bool:
        """Leave a generation scope and report whether cancelled memory cleanup is needed."""

        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            if self._active_count > 0 or self._subprocesses:
                return False
            was_cancelled = self._cancel_event.is_set()
            self._cancel_event.clear()
            return was_cancelled

    def request_cancel(self, *, subprocess_only: bool = False) -> bool:
        """Set cancellation and terminate tracked subprocess workers."""

        with self._lock:
            subprocesses = tuple(self._subprocesses)
            had_active_work = bool(subprocesses) if subprocess_only else (
                self._active_count > 0 or bool(subprocesses)
            )
            if had_active_work:
                self._cancel_event.set()
            elif not subprocess_only or self._active_count == 0:
                self._cancel_event.clear()

        for process in subprocesses:
            terminate_generation_process(process)
        return had_active_work

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._cancel_event.is_set()

    def is_active(self) -> bool:
        """Return whether any generation scope or subprocess is currently active."""

        with self._lock:
            return self._active_count > 0 or bool(self._subprocesses)

    def register_subprocess(self, process: Any) -> None:
        """Track a subprocess so cancel can terminate isolated generation."""

        with self._lock:
            self._subprocesses.add(process)

    def unregister_subprocess(self, process: Any) -> bool:
        """Stop tracking a subprocess and report whether cancelled cleanup is needed."""

        with self._lock:
            self._subprocesses.discard(process)
            if self._active_count > 0 or self._subprocesses:
                return False
            was_cancelled = self._cancel_event.is_set()
            self._cancel_event.clear()
            return was_cancelled


_STATE = _GenerationCancellationState()


@contextmanager
def generation_cancel_scope() -> Iterator[None]:
    """Scope one generation or batch run under cooperative cancellation."""

    _STATE.begin()
    try:
        yield
    finally:
        if _STATE.end():
            cleanup_runtime_memory()


def request_generation_cancel(*, subprocess_only: bool = False) -> bool:
    """Request cancellation for active generation work."""

    return _STATE.request_cancel(subprocess_only=subprocess_only)


def is_generation_cancelled() -> bool:
    """Return whether the current generation run has been cancelled."""

    return _STATE.is_cancelled()


def is_generation_active() -> bool:
    """Return whether generation work is currently registered as active."""

    return _STATE.is_active()


def check_generation_cancelled() -> None:
    """Raise ``GenerationCancelled`` if cancellation has been requested."""

    if _STATE.is_cancelled():
        raise GenerationCancelled(CANCEL_MESSAGE)


def register_generation_subprocess(process: Any) -> None:
    """Register an isolated generation worker process for cancellation."""

    _STATE.register_subprocess(process)


def unregister_generation_subprocess(process: Any) -> None:
    """Unregister an isolated generation worker process."""

    if _STATE.unregister_subprocess(process):
        cleanup_runtime_memory()


def cleanup_runtime_memory() -> None:
    """Best-effort release of Python and accelerator allocator caches."""

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        _empty_torch_cache(torch)
    mlx_core = sys.modules.get("mlx.core")
    if mlx_core is not None and hasattr(mlx_core, "clear_cache"):
        try:
            mlx_core.clear_cache()
        except Exception as exc:  # pragma: no cover - defensive accelerator cleanup.
            logger.debug("[generation_cancel] MLX cache cleanup skipped: {}", exc)


def _empty_torch_cache(torch: Any) -> None:
    """Release known torch accelerator caches when available."""

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:  # pragma: no cover - defensive accelerator cleanup.
        logger.debug("[generation_cancel] CUDA cache cleanup skipped: {}", exc)

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception as exc:  # pragma: no cover - defensive accelerator cleanup.
        logger.debug("[generation_cancel] XPU cache cleanup skipped: {}", exc)

    try:
        if (
            hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
    except Exception as exc:  # pragma: no cover - defensive accelerator cleanup.
        logger.debug("[generation_cancel] MPS cache cleanup skipped: {}", exc)
