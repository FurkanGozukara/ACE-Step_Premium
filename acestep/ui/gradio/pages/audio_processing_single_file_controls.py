"""Single-file controls for the Audio Processing page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.interfaces.source_audio_preview import (
    AUDIO_PROCESSING_AFTER_PREVIEW_ELEM_ID,
    AUDIO_PROCESSING_BEFORE_PREVIEW_ELEM_ID,
    AUDIO_PROCESSING_OUTPUT_PREVIEW_ELEM_ID,
    AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
    AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
    AUDIO_PROCESSING_UPLOAD_PREVIEW_ELEM_ID,
)


def add_single_file_controls(controls: dict[str, Any]) -> None:
    """Add single audio/video upload and output controls."""

    controls["ap_disable_upload_preview"] = gr.Checkbox(
        label="Disable upload preview",
        value=False,
        info=(
            "Use for very large videos or containers like multi-GB MKV files. "
            "When enabled, Gradio will not render the uploaded media preview, "
            "avoiding slow browser/Gradio post-processing such as MKV-to-MP4 "
            "preview conversion. Processing still uses the original uploaded file."
        ),
    )
    gr.Markdown("### Single Audio or Video")
    controls["ap_single_file"] = gr.File(
        label="Upload Audio or Video",
        file_count="multiple",
        type="filepath",
        file_types=[
            ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus",
            ".mp4", ".mov", ".mkv", ".webm", ".avi",
        ],
        elem_id="acestep-audio-processing-single-upload",
        key="audio_processing_single_upload",
        preserved_by_key=[],
    )
    with gr.Row():
        controls["ap_upload_audio_preview"] = gr.Audio(
            label="Uploaded Audio",
            type="filepath",
            interactive=True,
            editable=True,
            visible=False,
            elem_id=AUDIO_PROCESSING_UPLOAD_PREVIEW_ELEM_ID,
            elem_classes=AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
            waveform_options=AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
        )
        controls["ap_upload_video_preview"] = gr.Video(
            label="Uploaded Video",
            interactive=False,
            visible=False,
            elem_classes=["ace-video-preview"],
        )
    with gr.Row(equal_height=True):
        controls["ap_preview_btn"] = gr.Button(
            "Preview 60s",
            variant="secondary",
            size="lg",
            elem_classes=["action-btn", "action-btn-preview"],
        )
        controls["ap_open_outputs_folder_btn"] = gr.Button(
            "Open Outputs Folder",
            variant="secondary",
            size="lg",
        )
    with gr.Row(equal_height=False, elem_classes=["ace-audio-processing-primary-row"]):
        controls["ap_process_btn"] = gr.Button(
            "Process File",
            variant="primary",
            size="lg",
            scale=1,
            elem_classes=[
                "action-btn",
                "action-btn-generate",
                "action-btn-audio-processing-main",
            ],
        )
        controls["ap_cancel_processing_btn"] = gr.Button(
            "Cancel",
            variant="stop",
            size="lg",
            scale=1,
            elem_classes=[
                "action-btn",
                "action-btn-cancel",
                "action-btn-audio-processing-main",
            ],
        )
    controls["ap_cancel_confirmed_state"] = gr.State(value=False)
    with gr.Row():
        controls["ap_preview_before_audio"] = gr.Audio(
            label="Preview Before",
            type="filepath",
            interactive=False,
            elem_id=AUDIO_PROCESSING_BEFORE_PREVIEW_ELEM_ID,
            elem_classes=AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
            waveform_options=AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
        )
        controls["ap_preview_after_audio"] = gr.Audio(
            label="Preview After",
            type="filepath",
            interactive=False,
            elem_id=AUDIO_PROCESSING_AFTER_PREVIEW_ELEM_ID,
            elem_classes=AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
            waveform_options=AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
        )
    controls["ap_output_audio"] = gr.Audio(
        label="Processed Audio",
        type="filepath",
        interactive=False,
        elem_id=AUDIO_PROCESSING_OUTPUT_PREVIEW_ELEM_ID,
        elem_classes=AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
        waveform_options=AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
    )
    controls["ap_output_video"] = gr.Video(
        label="Processed Video",
        interactive=False,
        visible=False,
        elem_classes=["ace-video-preview"],
    )
    controls["ap_spectrogram"] = gr.Plot(label="Before / After Spectrogram")
    controls["ap_single_files"] = gr.File(
        label="Saved Files",
        file_count="multiple",
        interactive=False,
        visible=False,
    )
    controls["ap_single_status"] = gr.Markdown(
        "Audio processing ready.",
        elem_classes=["ace-status-scroll-10"],
    )
