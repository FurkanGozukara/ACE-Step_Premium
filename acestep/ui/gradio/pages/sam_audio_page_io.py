"""Input, preview, and output controls for the SAM Audio Segment page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.interfaces.source_audio_preview import (
    SAM_AUDIO_PREVIEW_ELEM_CLASSES,
    SAM_RESIDUAL_AUDIO_PREVIEW_ELEM_ID,
    SAM_TARGET_AUDIO_PREVIEW_ELEM_ID,
    SAM_UPLOAD_AUDIO_PREVIEW_ELEM_ID,
    TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
)

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
                    file_count="multiple",
                    type="filepath",
                    file_types=MEDIA_FILE_TYPES,
                    elem_id="acestep-sam-single-upload",
                    key="sam_single_upload",
                    preserved_by_key=[],
                )
            with gr.Column(scale=1):
                controls["sam_visual_mask_file"] = gr.File(
                    label="Visual Mask Video",
                    file_count="multiple",
                    type="filepath",
                    file_types=MASK_VIDEO_FILE_TYPES,
                    interactive=False,
                    elem_id="acestep-sam-visual-mask-upload",
                    key="sam_visual_mask_upload",
                    preserved_by_key=[],
                )
                controls["sam_visual_mask_video_preview"] = gr.Video(
                    label="Visual Mask Video Preview",
                    interactive=False,
                    visible=False,
                    elem_classes=["ace-video-preview"],
                )
        with gr.Row():
            controls["sam_upload_audio_preview"] = gr.Audio(
                label="Uploaded Audio",
                type="filepath",
                sources=["upload"],
                interactive=True,
                editable=True,
                visible=False,
                elem_id=SAM_UPLOAD_AUDIO_PREVIEW_ELEM_ID,
                elem_classes=SAM_AUDIO_PREVIEW_ELEM_CLASSES,
                waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
            )
            controls["sam_upload_video_preview"] = gr.Video(
                label="Uploaded Video",
                interactive=False,
                visible=False,
                elem_classes=["ace-video-preview"],
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
                elem_id=SAM_TARGET_AUDIO_PREVIEW_ELEM_ID,
                elem_classes=SAM_AUDIO_PREVIEW_ELEM_CLASSES,
                waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
            )
            controls["sam_residual_audio"] = gr.Audio(
                label="Remaining Audio",
                type="filepath",
                interactive=False,
                elem_id=SAM_RESIDUAL_AUDIO_PREVIEW_ELEM_ID,
                elem_classes=SAM_AUDIO_PREVIEW_ELEM_CLASSES,
                waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
            )
        controls["sam_target_video"] = gr.Video(
            label="Extracted Video",
            interactive=False,
            visible=False,
            elem_classes=["ace-video-preview"],
        )
        controls["sam_single_files"] = gr.File(
            label="Saved Files",
            file_count="multiple",
            interactive=False,
            visible=False,
        )
        controls["sam_single_status"] = gr.Markdown(
            "SAM Audio ready.",
            elem_classes=["ace-status-scroll-10"],
        )
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
        with gr.Row():
            controls["sam_batch_recursive"] = gr.Checkbox(
                label="Include subfolders",
                value=False,
            )
            controls["sam_batch_save_output_only"] = gr.Checkbox(
                label="Save only output",
                value=False,
                info="Save only the extracted audio in the output folder.",
            )
            controls["sam_batch_overwrite_existing"] = gr.Checkbox(
                label="Overwrite Existing Files",
                value=False,
                info="Replace files with the same target name instead of adding _extractN.",
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
            lines=10,
            max_lines=10,
            interactive=False,
        )
        controls["sam_batch_cancel_confirmed_state"] = gr.State(value=False)
