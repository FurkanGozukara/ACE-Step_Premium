"""Console progress callback for isolated generation workers."""

from __future__ import annotations

from typing import Any


class WorkerConsoleProgress:
    """Print worker progress updates in a subprocess-friendly format."""

    def __init__(self) -> None:
        """Initialize the duplicate-message filter."""

        self._last_message = ""

    def __call__(self, value: Any = None, desc: Any = None, *args: Any, **kwargs: Any) -> None:
        """Print a progress update compatible with Gradio progress calls."""

        description = desc if desc is not None else _first_description(args, kwargs)
        message = _format_progress_message(value, description)
        if message == self._last_message:
            return
        self._last_message = message
        print(message, flush=True)


def _first_description(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Return the first positional or keyword description value."""

    if args:
        return args[0]
    return kwargs.get("desc")


def _format_progress_message(value: Any, description: Any) -> str:
    """Format one console progress line."""

    parts = ["[Worker progress]"]
    percent = _format_percent(value)
    if percent:
        parts.append(percent)
    if description:
        parts.append(str(description))
    return " ".join(parts)


def _format_percent(value: Any) -> str:
    """Return a percentage string for numeric progress values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if 0.0 <= numeric <= 1.0:
        numeric *= 100.0
    return f"{numeric:.0f}%"


worker_console_progress = WorkerConsoleProgress()
