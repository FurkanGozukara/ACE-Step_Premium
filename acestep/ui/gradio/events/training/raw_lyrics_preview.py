"""Raw-lyrics preview helpers for the training dataset UI."""

from __future__ import annotations

from typing import Any

import gradio as gr


def raw_lyrics_preview_update(raw_lyrics: Any, has_raw_lyrics: bool) -> Any:
    """Return one textbox update containing raw lyric text and visibility."""

    has_raw = bool(has_raw_lyrics)
    raw_text = str(raw_lyrics or "") if has_raw else ""
    return gr.update(value=raw_text, visible=has_raw)
