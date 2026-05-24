"""Progress text helpers for dataset auto-label batches."""

from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator


ProgressCallback = Callable[[str], None]
TimeFn = Callable[[], float]


@dataclass
class LabelProgressTracker:
    """Build auto-label progress messages with speed and ETA details."""

    total: int
    time_fn: TimeFn = time.monotonic
    started_at: float = field(init=False)
    item_started_at: float | None = field(default=None, init=False)
    last_seconds: float | None = field(default=None, init=False)
    completed_items: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Record the batch start time."""

        self.started_at = self.time_fn()

    def begin_item(self) -> None:
        """Record the start time for the current sample."""

        self.item_started_at = self.time_fn()

    def complete_item(self) -> None:
        """Record completion timing for the current sample."""

        now = self.time_fn()
        if self.item_started_at is not None:
            self.last_seconds = max(0.0, now - self.item_started_at)
        self.completed_items += 1

    def start_message(
        self,
        position: int,
        labeled_count: int,
        left_count: int,
        filename: str,
    ) -> str:
        """Return the in-progress status for a sample."""

        return (
            f"Labeling {position}/{self.total}; labeled {labeled_count}/{self.total}; "
            f"left {left_count}: {filename} | {self._timing_summary(left_count)}"
        )

    def complete_message(
        self,
        position: int,
        labeled_count: int,
        left_count: int,
        filename: str,
    ) -> str:
        """Return the completed status for a sample."""

        return (
            f"Labeling {position}/{self.total} complete; "
            f"labeled {labeled_count}/{self.total}; left {left_count}: {filename} | "
            f"{self._timing_summary(left_count)}"
        )

    def _timing_summary(self, left_count: int) -> str:
        """Return elapsed time, last item duration, speed, and ETA text."""

        elapsed = max(0.0, self.time_fn() - self.started_at)
        parts = [f"elapsed {_format_duration(elapsed)}"]
        if self.last_seconds is not None:
            parts.append(f"last {_format_duration(self.last_seconds)}")
        if self.completed_items <= 0:
            parts.append("ETA calculating")
            return "; ".join(parts)

        average_seconds = elapsed / self.completed_items if elapsed > 0 else 0.0
        speed_per_minute = (self.completed_items / elapsed * 60.0) if elapsed > 0 else 0.0
        parts.append(f"avg {_format_duration(average_seconds)}/file")
        parts.append(f"speed {speed_per_minute:.2f}/min")
        parts.append(f"ETA {_format_duration(average_seconds * max(left_count, 0))}")
        return "; ".join(parts)


@contextmanager
def replay_progress_after_llm_load(
    llm_handler: object,
    progress_callback: ProgressCallback | None,
    message: str,
) -> Iterator[None]:
    """Temporarily print the current progress message after LLM GPU loads."""

    if llm_handler is None or progress_callback is None:
        yield
        return

    attr_name = "_post_load_status_message"
    sentinel = object()
    previous = getattr(llm_handler, attr_name, sentinel)
    setattr(llm_handler, attr_name, message)
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(llm_handler, attr_name)
        else:
            setattr(llm_handler, attr_name, previous)


@contextmanager
def progress_heartbeat(
    progress_callback: ProgressCallback | None,
    message: str,
    *,
    interval_seconds: float = 5.0,
) -> Iterator[None]:
    """Emit periodic progress while a blocking model call is running."""

    if progress_callback is None or interval_seconds <= 0:
        yield
        return

    stop_event = threading.Event()
    started_at = time.monotonic()

    def emit_until_stopped() -> None:
        """Emit elapsed-time progress until the guarded operation finishes."""

        while not stop_event.wait(interval_seconds):
            elapsed = _format_duration(time.monotonic() - started_at)
            try:
                progress_callback(f"{message} | elapsed {elapsed}")
            except Exception:
                pass

    thread = threading.Thread(
        target=emit_until_stopped,
        name="auto-label-progress-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=min(interval_seconds, 1.0))


def _format_duration(seconds: float) -> str:
    """Format seconds as compact HH:MM:SS or MM:SS text."""

    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"
