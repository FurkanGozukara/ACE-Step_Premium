"""Generated-song library page for the premium Gradio shell."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.generated_library import TABLE_HEADERS
from acestep.ui.gradio.interfaces.source_audio_preview import (
    TRIM_AUDIO_PREVIEW_ELEM_CLASSES,
    TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
)


def create_library_page() -> dict[str, Any]:
    """Build the generated-song library tab."""

    library_state = gr.State(value=[])
    library_filtered_state = gr.State(value=[])
    library_page_state = gr.State(value=1)
    library_timezone_state = gr.State(value=None)
    with gr.Row(equal_height=True):
        refresh_library_btn = gr.Button(
            "Refresh Library",
            variant="secondary",
            size="lg",
            elem_classes=["action-btn", "action-btn-open"],
        )
        library_selector = gr.Dropdown(
            label="Generated Songs by Day",
            choices=[],
            value=None,
            interactive=True,
            scale=5,
        )
    with gr.Row(equal_height=True):
        library_search_query = gr.Textbox(
            label="Search Songs",
            value="",
            placeholder="rock* vocals",
            interactive=True,
            scale=5,
        )
        library_prev_page_btn = gr.Button(
            "Previous",
            variant="secondary",
            size="sm",
            interactive=False,
            scale=1,
            min_width=100,
        )
        library_page_status = gr.Markdown("No songs loaded.")
        library_next_page_btn = gr.Button(
            "Next",
            variant="secondary",
            size="sm",
            interactive=False,
            scale=1,
            min_width=100,
        )

    with gr.Row():
        with gr.Column(scale=5, min_width=420):
            library_table = gr.Dataframe(
                headers=TABLE_HEADERS,
                value=[],
                interactive=False,
                wrap=True,
                label="Recent Generations",
            )
            library_details = gr.Markdown("No generated songs found yet.")
        with gr.Column(scale=4, min_width=360):
            library_audio = gr.Audio(
                label="Selected Song",
                type="filepath",
                interactive=False,
                buttons=["download"],
                elem_classes=TRIM_AUDIO_PREVIEW_ELEM_CLASSES,
                waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
            )
            library_lyrics = gr.Textbox(
                label="Lyrics",
                interactive=False,
                lines=14,
                max_lines=22,
                buttons=["copy"],
            )
            library_metadata = gr.Textbox(
                label="Full Metadata JSON",
                value="",
                interactive=False,
                lines=14,
                max_lines=22,
                buttons=["copy"],
            )

    return {
        "library_state": library_state,
        "library_filtered_state": library_filtered_state,
        "library_page_state": library_page_state,
        "library_timezone_state": library_timezone_state,
        "refresh_library_btn": refresh_library_btn,
        "library_selector": library_selector,
        "library_search_query": library_search_query,
        "library_prev_page_btn": library_prev_page_btn,
        "library_page_status": library_page_status,
        "library_next_page_btn": library_next_page_btn,
        "library_table": library_table,
        "library_audio": library_audio,
        "library_details": library_details,
        "library_lyrics": library_lyrics,
        "library_metadata": library_metadata,
    }
