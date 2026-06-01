"""Audio Processing page for manual and generated-song processing."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_trim_settings import (
    AUTO_EDITOR_MARGIN_DEFAULT_SECONDS,
    AUTO_EDITOR_MARGIN_MAX_SECONDS,
    AUTO_EDITOR_MARGIN_MIN_SECONDS,
    AUTO_EDITOR_MINCLIP_DEFAULT,
    AUTO_EDITOR_MINCUT_DEFAULT,
    AUTO_EDITOR_SMOOTH_MAX,
    AUTO_EDITOR_SMOOTH_MIN,
    AUTO_EDITOR_THRESHOLD_DEFAULT_DB,
    AUTO_EDITOR_THRESHOLD_MAX_DB,
    AUTO_EDITOR_THRESHOLD_MIN_DB,
)
from acestep.audio_processing.presets import (
    DEFAULT_STAGE_VALUES,
    OUTPUT_FORMAT_CHOICES,
    PRESET_VALUES,
    STAGE_KEYS,
    STAGE_LABELS,
)


def create_audio_processing_page() -> dict[str, Any]:
    """Build the Audio Processing tab controls."""

    controls: dict[str, Any] = {}
    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Audio Processing</span>
              <h2>Enhance, pre-master, and optionally post-process generated songs.</h2>
            </section>
            """
        )
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            with gr.Row():
                controls["ap_auto_postprocess"] = gr.Checkbox(
                    label="Apply automatically to generated songs",
                    value=False,
                    info="Runs after Generate Song and Advanced generation save their original audio.",
                )
                controls["ap_preserve_original"] = gr.Checkbox(
                    label="Save original plus processed song",
                    value=True,
                    info="Enabled by default so generated originals remain available.",
                )
                controls["ap_output_format"] = gr.Dropdown(
                    choices=OUTPUT_FORMAT_CHOICES,
                    value="wav",
                    label="Processed Output",
                )
                controls["ap_builtin_preset"] = gr.Dropdown(
                    choices=list(PRESET_VALUES.keys()),
                    value="Generic AI",
                    label="Processing Preset",
                )
            with gr.Row(equal_height=True):
                controls["ap_trim_empty_output"] = gr.Checkbox(
                    label="Auto-Editor trim silent sections",
                    value=False,
                    info=(
                        "Optional. Uses auto-editor to trim out silent sections "
                        "after audio processing."
                    ),
                    scale=2,
                    min_width=220,
                )
                controls["ap_trim_threshold_db"] = gr.Slider(
                    minimum=AUTO_EDITOR_THRESHOLD_MIN_DB,
                    maximum=AUTO_EDITOR_THRESHOLD_MAX_DB,
                    step=1.0,
                    value=AUTO_EDITOR_THRESHOLD_DEFAULT_DB,
                    label="Auto-Editor threshold (dB)",
                    info=(
                        "Default -40 dB, matching Auto Encode & Shorten."
                    ),
                    scale=2,
                    min_width=180,
                )
                controls["ap_trim_margin_seconds"] = gr.Slider(
                    minimum=AUTO_EDITOR_MARGIN_MIN_SECONDS,
                    maximum=AUTO_EDITOR_MARGIN_MAX_SECONDS,
                    step=0.1,
                    value=AUTO_EDITOR_MARGIN_DEFAULT_SECONDS,
                    label="Auto-Editor margin (s)",
                    info="Keeps this much audio before and after non-silent sections.",
                    scale=2,
                    min_width=180,
                )
                controls["ap_trim_mincut"] = gr.Slider(
                    minimum=AUTO_EDITOR_SMOOTH_MIN,
                    maximum=AUTO_EDITOR_SMOOTH_MAX,
                    step=1,
                    value=AUTO_EDITOR_MINCUT_DEFAULT,
                    label="Auto-Editor mincut",
                    info="Default 20, matching Auto Encode & Shorten.",
                    scale=2,
                    min_width=160,
                )
                controls["ap_trim_minclip"] = gr.Slider(
                    minimum=AUTO_EDITOR_SMOOTH_MIN,
                    maximum=AUTO_EDITOR_SMOOTH_MAX,
                    step=1,
                    value=AUTO_EDITOR_MINCLIP_DEFAULT,
                    label="Auto-Editor minclip",
                    info="Default 4, matching Auto Encode & Shorten.",
                    scale=2,
                    min_width=160,
                )

        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            gr.Markdown("### Single Audio or Video")
            controls["ap_single_file"] = gr.File(
                label="Upload Audio or Video",
                file_count="single",
                type="filepath",
                file_types=[
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
                ],
            )
            with gr.Row():
                controls["ap_upload_audio_preview"] = gr.Audio(
                    label="Uploaded Audio",
                    type="filepath",
                    interactive=False,
                    visible=False,
                )
                controls["ap_upload_video_preview"] = gr.Video(
                    label="Uploaded Video",
                    interactive=False,
                    visible=False,
                )
            with gr.Row(equal_height=True):
                controls["ap_preview_btn"] = gr.Button(
                    "Preview 60s",
                    variant="secondary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-preview"],
                )
                controls["ap_process_btn"] = gr.Button(
                    "Process File",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-generate"],
                )
            with gr.Row():
                controls["ap_preview_before_audio"] = gr.Audio(
                    label="Preview Before",
                    type="filepath",
                    interactive=False,
                )
                controls["ap_preview_after_audio"] = gr.Audio(
                    label="Preview After",
                    type="filepath",
                    interactive=False,
                )
            controls["ap_output_audio"] = gr.Audio(
                label="Processed Audio",
                type="filepath",
                interactive=False,
            )
            controls["ap_output_video"] = gr.Video(
                label="Processed Video",
                interactive=False,
                visible=False,
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

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("### Audio Enhancement")
                for key in STAGE_KEYS[:6]:
                    _add_stage_control(controls, key)
            with gr.Column(scale=1):
                gr.Markdown("### Pre-Mastering")
                for key in STAGE_KEYS[6:]:
                    _add_stage_control(controls, key)

        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            gr.Markdown("### Batch Folder Processing")
            with gr.Row():
                with gr.Column(scale=3):
                    controls["ap_batch_input_folder"] = gr.Textbox(label="Input Folder")
                    controls["ap_batch_input_browse_btn"] = gr.Button("Browse Input Folder")
                with gr.Column(scale=3):
                    controls["ap_batch_output_folder"] = gr.Textbox(
                        label="Output Folder",
                        info="Optional. Blank values save under the ACE-Step outputs folder.",
                    )
                    controls["ap_batch_output_browse_btn"] = gr.Button("Browse Output Folder")
            controls["ap_batch_recursive"] = gr.Checkbox(
                label="Include subfolders",
                value=False,
            )
            controls["ap_batch_process_btn"] = gr.Button(
                "Process Batch Folder",
                variant="primary",
                size="lg",
                elem_classes=["action-btn", "action-btn-generate"],
            )
            controls["ap_batch_files"] = gr.File(
                label="Batch Saved Files",
                file_count="multiple",
                interactive=False,
                visible=False,
            )
            controls["ap_batch_status"] = gr.Textbox(
                label="Batch Status",
                value="Select a folder and process supported audio/video files.",
                lines=10,
                max_lines=10,
                interactive=False,
            )
    return controls


def _add_stage_control(controls: dict[str, Any], key: str) -> None:
    """Add one enabled checkbox and value slider to the page controls."""

    minimum, maximum, step = _slider_bounds(key)
    with gr.Row(equal_height=True):
        controls[f"ap_{key}_enabled"] = gr.Checkbox(
            label=STAGE_LABELS[key],
            value=True,
            scale=1,
        )
        controls[f"ap_{key}"] = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            step=step,
            value=DEFAULT_STAGE_VALUES[key],
            label=STAGE_LABELS[key],
            scale=4,
        )


def _slider_bounds(key: str) -> tuple[float, float, float]:
    """Return slider bounds for an audio-processing stage."""

    if key == "stereo":
        return 1.0, 2.0, 0.05
    if key == "lufs":
        return -24.0, -8.0, 0.5
    return 0.0, 1.0, 0.05
