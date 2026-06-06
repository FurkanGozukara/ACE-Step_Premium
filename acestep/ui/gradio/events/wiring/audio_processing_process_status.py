"""Status helpers for Audio Processing Gradio actions."""

from __future__ import annotations

from typing import Any

from acestep.audio_processing.runs import DEFAULT_RESULTS_DIR
from acestep.ui.gradio.events.local_path_dialogs import open_folder_path


def make_process_log_callback(process_log: list[str], progress: Any = None):
    """Return a callback that records processing messages for the status panel."""

    def _callback(progress_value: Any = None, label: Any = None) -> None:
        text = str(label if label is not None else progress_value or "").strip()
        if text:
            progress_fraction = _progress_fraction(progress_value)
            _update_gradio_progress(progress, progress_fraction, text)
            process_log.append(_format_process_log_text(text, progress_fraction))

    return _callback


def with_process_log(markdown: str, process_log: list[str]) -> str:
    """Append process log lines to a status markdown block."""

    if not process_log:
        return markdown
    lines = "\n".join(f"- {line}" for line in process_log[-80:])
    return f"{markdown}\n\n### Process Log\n{lines}"


def open_audio_processing_outputs_folder() -> str:
    """Open the default Audio Processing outputs folder."""

    return open_folder_path(str(DEFAULT_RESULTS_DIR))


def _progress_fraction(value: Any) -> float | None:
    """Return a normalized progress fraction when the callback carries one."""

    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction != fraction:
        return None
    return max(0.0, min(1.0, fraction))


def _format_process_log_text(text: str, progress_fraction: float | None) -> str:
    """Prefix plain process-log messages with a parsed percentage."""

    if progress_fraction is None or "%" in text:
        return text
    return f"{progress_fraction * 100:.1f}% - {text}"


def _update_gradio_progress(progress: Any, progress_fraction: float | None, text: str) -> None:
    """Update Gradio progress when running inside a Gradio event."""

    if progress is None or progress_fraction is None:
        return
    try:
        progress(progress_fraction, desc=text)
    except Exception:
        return
