"""Status-log helpers for streaming SAM-Audio progress into Gradio."""

from __future__ import annotations

from queue import Empty, Queue
from typing import Any

ProgressEvent = tuple[float, str]


class SamAudioStatusLog:
    """Collect SAM-Audio progress events for Gradio status updates."""

    def __init__(self, progress: Any | None, *, max_lines: int = 80) -> None:
        self.progress = progress
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.queue: Queue[ProgressEvent] = Queue()

    def callback(self, fraction: float, message: str) -> None:
        """Queue one progress event from a worker thread."""

        self.queue.put((float(fraction), str(message)))

    def drain(self) -> bool:
        """Move queued progress events into the rendered log."""

        changed = False
        while True:
            try:
                fraction, message = self.queue.get_nowait()
            except Empty:
                break
            if self.progress is not None:
                self.progress(fraction, desc=message)
            self.lines.append(_format_line(fraction, message))
            self.lines = self.lines[-self.max_lines :]
            changed = True
        return changed

    def render(self, heading: str = "SAM-Audio running...") -> str:
        """Return status markdown with recent progress lines."""

        if not self.lines:
            return heading
        return heading + "\n\n### Run Log\n" + "\n".join(self.lines)

    def append_to_status(self, status: str) -> str:
        """Append the captured run log to final status markdown."""

        self.drain()
        if not self.lines:
            return status
        return status + "\n\n### Run Log\n" + "\n".join(self.lines)


def _format_line(fraction: float, message: str) -> str:
    """Return one compact markdown log line."""

    percent = max(0, min(100, round(float(fraction) * 100)))
    return f"- {percent:>3}% - {message}"
