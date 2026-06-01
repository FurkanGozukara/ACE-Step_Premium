"""Input, preview, and output controls for the SAM Audio Segment page."""

from __future__ import annotations

from typing import Any

import gradio as gr

MEDIA_FILE_TYPES = [
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
]

MASK_VIDEO_FILE_TYPES = [".mp4", ".mov", ".mkv", ".webm", ".avi"]


def add_single_file_controls(controls: dict[str, Any]) -> None:
    """Add single-file upload, preview, and output controls."""

    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        gr.Markdown("### Single Audio or Video")
        with gr.Row():
            with gr.Column(scale=1):
                controls["sam_single_file"] = gr.File(
                    label="Upload Audio or Video",
                    file_count="single",
                    type="filepath",
                    file_types=MEDIA_FILE_TYPES,
                )
            with gr.Column(scale=1):
                controls["sam_visual_mask_file"] = gr.File(
                    label="Visual Mask Video",
                    file_count="single",
                    type="filepath",
                    file_types=MASK_VIDEO_FILE_TYPES,
                    interactive=False,
                )
        with gr.Row():
            controls["sam_upload_audio_preview"] = gr.Audio(
                label="Uploaded Audio",
                type="filepath",
                interactive=False,
                visible=False,
            )
            controls["sam_upload_video_preview"] = gr.Video(
                label="Uploaded Video",
                interactive=False,
                visible=False,
            )
        with gr.Row(equal_height=False, elem_classes=["ace-generate-action-row"]):
            controls["sam_process_btn"] = gr.Button(
                "Segment File",
                variant="primary",
                size="lg",
                scale=2,
                elem_classes=["action-btn", "action-btn-generate"],
            )
            controls["sam_cancel_btn"] = gr.Button(
                "Cancel",
                variant="stop",
                size="lg",
                scale=1,
                elem_classes=["action-btn", "action-btn-cancel"],
            )
            controls["sam_open_outputs_btn"] = gr.Button(
                "Open Outputs Folder",
                size="lg",
                scale=1,
                elem_classes=["action-btn", "action-btn-open"],
            )
        with gr.Row():
            controls["sam_target_audio"] = gr.Audio(
                label="Extracted Audio",
                type="filepath",
                interactive=False,
            )
            controls["sam_residual_audio"] = gr.Audio(
                label="Remaining Audio",
                type="filepath",
                interactive=False,
            )
        controls["sam_target_video"] = gr.Video(
            label="Extracted Video",
            interactive=False,
            visible=False,
        )
        controls["sam_single_files"] = gr.File(
            label="Saved Files",
            file_count="multiple",
            interactive=False,
            visible=False,
        )
        controls["sam_single_status"] = gr.Markdown("SAM Audio ready.")
        controls["sam_cancel_confirmed_state"] = gr.State(value=False)


def add_batch_controls(controls: dict[str, Any]) -> None:
    """Add batch-folder controls."""

    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        gr.Markdown("### Batch Folder Processing")
        with gr.Row():
            with gr.Column(scale=3):
                controls["sam_batch_input_folder"] = gr.Textbox(label="Input Folder")
                controls["sam_batch_input_browse_btn"] = gr.Button("Browse Input Folder")
            with gr.Column(scale=3):
                controls["sam_batch_output_folder"] = gr.Textbox(label="Output Folder")
                controls["sam_batch_output_browse_btn"] = gr.Button("Browse Output Folder")
        controls["sam_batch_recursive"] = gr.Checkbox(
            label="Include subfolders",
            value=False,
        )
        with gr.Row(equal_height=False, elem_classes=["ace-generate-action-row"]):
            controls["sam_batch_process_btn"] = gr.Button(
                "Segment Batch Folder",
                variant="primary",
                size="lg",
                scale=2,
                elem_classes=["action-btn", "action-btn-generate"],
            )
            controls["sam_batch_cancel_btn"] = gr.Button(
                "Cancel Batch",
                variant="stop",
                size="lg",
                scale=1,
                elem_classes=["action-btn", "action-btn-cancel"],
            )
        controls["sam_batch_files"] = gr.File(
            label="Batch Saved Files",
            file_count="multiple",
            interactive=False,
            visible=False,
        )
        controls["sam_batch_status"] = gr.Textbox(
            label="Batch Status",
            value="Select a folder and process supported audio/video files.",
            lines=16,
            interactive=False,
        )
        controls["sam_batch_cancel_confirmed_state"] = gr.State(value=False)
