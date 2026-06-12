"""Shared Gradio output updates for Audio Processing results."""

from __future__ import annotations

from typing import Any

import gradio as gr


def hidden_output(value: Any = None) -> dict[str, Any]:
    """Return a hidden Gradio update, optionally clearing its value."""

    return gr.update(value=value, visible=False)


def visible_output(value: Any) -> dict[str, Any]:
    """Return a visible Gradio update with the provided value."""

    return gr.update(value=value, visible=True)


def visible_if_present(value: Any) -> dict[str, Any]:
    """Return a visible Gradio update only when the value is present."""

    return gr.update(value=value, visible=bool(value))
