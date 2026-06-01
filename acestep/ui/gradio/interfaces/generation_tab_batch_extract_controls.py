"""Batch Extract controls for the generation tab."""

from typing import Any

import gradio as gr


def build_batch_extract_controls() -> dict[str, Any]:
    """Create folder controls for running Extract over every audio file in a folder."""

    with gr.Accordion("Batch Extract", open=True) as batch_extract_group:
        with gr.Row():
            batch_extract_input_folder = gr.Textbox(
                label="Batch Extract Input Folder",
                placeholder="Folder containing source audio files",
                scale=4,
            )
            batch_extract_input_browse_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            batch_extract_output_folder = gr.Textbox(
                label="Batch Extract Output Folder *",
                placeholder="Required folder where extracted files will be saved",
                scale=4,
            )
            batch_extract_output_browse_btn = gr.Button("Browse", scale=1)
        with gr.Row(equal_height=False, elem_classes=["ace-generate-action-row"]):
            batch_extract_btn = gr.Button(
                "Batch Extract",
                variant="primary",
                size="lg",
                scale=2,
                elem_classes=["action-btn", "action-btn-generate"],
            )
            batch_extract_cancel_btn = gr.Button(
                "Cancel Batch Extract",
                variant="stop",
                size="lg",
                scale=1,
                elem_classes=["action-btn", "action-btn-cancel"],
            )
        batch_extract_status = gr.Textbox(
            label="Batch Extract Status",
            interactive=False,
            lines=10,
            max_lines=10,
        )
        batch_extract_cancel_confirmed_state = gr.State(value=False)

    return {
        "batch_extract_group": batch_extract_group,
        "batch_extract_input_folder": batch_extract_input_folder,
        "batch_extract_input_browse_btn": batch_extract_input_browse_btn,
        "batch_extract_output_folder": batch_extract_output_folder,
        "batch_extract_output_browse_btn": batch_extract_output_browse_btn,
        "batch_extract_btn": batch_extract_btn,
        "batch_extract_cancel_btn": batch_extract_cancel_btn,
        "batch_extract_status": batch_extract_status,
        "batch_extract_cancel_confirmed_state": batch_extract_cancel_confirmed_state,
    }
