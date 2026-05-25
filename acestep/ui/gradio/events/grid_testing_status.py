"""Status formatting helpers for Grid Testing."""

from __future__ import annotations

from typing import Any, Sequence

import gradio as gr
from loguru import logger


def emit_grid_status(status_lines: list[str], message: str) -> None:
    """Append and print a Grid Testing status line.

    Args:
        status_lines: Mutable status history.
        message: New status text.
    """

    status_lines.append(message)
    logger.info("[grid_testing] {}", message)
    print(f"[grid_testing] {message}", flush=True)


def render_grid_status(lines: Sequence[str]) -> str:
    """Return a compact status log for the Gradio textbox."""

    if len(lines) <= 80:
        return "\n".join(lines)
    return "\n".join(["... earlier messages omitted ...", *lines[-79:]])


def grid_files_update(paths: list[str]) -> Any:
    """Return a Gradio update for generated grid files."""

    return gr.update(value=paths, visible=bool(paths))
