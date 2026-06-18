"""Batch process controls for the generation tab."""

from typing import Any

import gradio as gr


def build_batch_extract_controls() -> dict[str, Any]:
    """Create folder controls for running the selected task over a folder."""

    with gr.Accordion("Batch Process", open=False) as batch_extract_group:
        with gr.Row():
            batch_extract_input_folder = gr.Textbox(
                label="Batch Process Input Folder",
                placeholder="Folder containing source audio files",
                scale=4,
            )
            batch_extract_input_browse_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            batch_extract_output_folder = gr.Textbox(
                label="Batch Process Output Folder *",
                placeholder="Required folder where processed files will be saved",
                scale=4,
            )
            batch_extract_output_browse_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            batch_extract_recursive = gr.Checkbox(
                label="Include subfolders",
                value=False,
            )
            batch_extract_save_output_only = gr.Checkbox(
                label="Save only output",
                value=False,
                info="Save only the extracted audio in the output folder.",
            )
        with gr.Row(equal_height=False, elem_classes=["ace-generate-action-row"]):
            batch_extract_btn = gr.Button(
                "Batch Process",
                variant="primary",
                size="lg",
                scale=2,
                elem_classes=["action-btn", "action-btn-generate"],
            )
            batch_extract_cancel_btn = gr.Button(
                "Cancel Batch Process",
                variant="stop",
                size="lg",
                scale=1,
                elem_classes=["action-btn", "action-btn-cancel"],
            )
        batch_extract_status = gr.Textbox(
            label="Batch Process Status",
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
        "batch_extract_recursive": batch_extract_recursive,
        "batch_extract_save_output_only": batch_extract_save_output_only,
        "batch_extract_btn": batch_extract_btn,
        "batch_extract_cancel_btn": batch_extract_cancel_btn,
        "batch_extract_status": batch_extract_status,
        "batch_extract_cancel_confirmed_state": batch_extract_cancel_confirmed_state,
    }
