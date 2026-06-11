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
from acestep.audio_processing.auto_editor_workflow import (
    AUTO_EDITOR_WORKFLOW_EXPORT_CHOICES,
    AUTO_EDITOR_WORKFLOW_EXPORT_KEY,
    AUTO_EDITOR_WORKFLOW_EXPORT_NONE,
)
from acestep.audio_processing.presets import (
    OUTPUT_FORMAT_CHOICES,
    PRESET_VALUES,
)
from acestep.ui.gradio.pages.audio_processing_diffpitcher_controls import (
    add_diffpitcher_controls,
)
from acestep.ui.gradio.pages.audio_processing_single_file_controls import (
    add_single_file_controls,
)
from acestep.ui.gradio.pages.audio_processing_stage_controls import add_stage_controls
from acestep.ui.gradio.pages.audio_processing_video_controls import (
    add_video_auto_editor_controls,
)


def create_audio_processing_page() -> dict[str, Any]:
    """Build the Audio Processing tab controls."""

    controls: dict[str, Any] = {}
    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            with gr.Row():
                controls["ap_auto_postprocess"] = gr.Checkbox(
                    label="Apply automatically to generated songs",
                    value=False,
                    info=(
                        "Runs after Generate Song and Advanced generation save "
                        "their original audio."
                    ),
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
                controls["ap_run_subprocess"] = gr.Checkbox(
                    label="Run as subprocess",
                    value=True,
                    info="Runs Process File in an isolated worker that can be cancelled.",
                )
                controls["ap_export_audio_only"] = gr.Checkbox(
                    label="Export Only Audio",
                    value=False,
                    info="For video inputs, save processed audio without producing video.",
                )
            with gr.Row(equal_height=True):
                controls["ap_trim_empty_output"] = gr.Checkbox(
                    label="Auto-Editor trim silent sections",
                    value=False,
                    info=(
                        "Runs Auto-Editor after processing. It keeps sections louder "
                        "than the threshold and cuts quiet gaps; ACE-Step Extract and "
                        "SAM Audio auto trim use these same trim behavior values."
                    ),
                    elem_id="acestep-ap-trim-empty-output",
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
                        "Source-audio level used to decide what is active. Less "
                        "negative is more aggressive: -15 dB cuts quiet residue, "
                        "-40 dB keeps more tails. Auto-Editor max is 0 dB."
                    ),
                    elem_id="acestep-ap-trim-threshold-db",
                    scale=2,
                    min_width=180,
                )
                controls["ap_trim_margin_seconds"] = gr.Slider(
                    minimum=AUTO_EDITOR_MARGIN_MIN_SECONDS,
                    maximum=AUTO_EDITOR_MARGIN_MAX_SECONDS,
                    step=0.1,
                    value=AUTO_EDITOR_MARGIN_DEFAULT_SECONDS,
                    label="Auto-Editor margin (s)",
                    info=(
                        "Extra audio kept before and after each kept section. "
                        "Example: 0.3 keeps 0.3 seconds around a detected vocal or "
                        "beat so the cut does not clip the attack or tail."
                    ),
                    elem_id="acestep-ap-trim-margin-seconds",
                    scale=2,
                    min_width=180,
                )
                controls["ap_trim_mincut"] = gr.Slider(
                    minimum=AUTO_EDITOR_SMOOTH_MIN,
                    maximum=AUTO_EDITOR_SMOOTH_MAX,
                    step=1,
                    value=AUTO_EDITOR_MINCUT_DEFAULT,
                    label="Auto-Editor mincut",
                    info=(
                        "Shortest quiet gap Auto-Editor may remove, in timeline "
                        "frames, usually 30 per second. Example: 20 is about 0.67s, "
                        "so a 0.3s pause stays but a 1s gap can be cut."
                    ),
                    elem_id="acestep-ap-trim-mincut",
                    scale=2,
                    min_width=160,
                )
                controls["ap_trim_minclip"] = gr.Slider(
                    minimum=AUTO_EDITOR_SMOOTH_MIN,
                    maximum=AUTO_EDITOR_SMOOTH_MAX,
                    step=1,
                    value=AUTO_EDITOR_MINCLIP_DEFAULT,
                    label="Auto-Editor minclip",
                    info=(
                        "Shortest active section Auto-Editor may keep, in timeline "
                        "frames. Example: 4 is about 0.13s and keeps quick consonants; "
                        "higher values can discard tiny noises but may remove short hits."
                    ),
                    elem_id="acestep-ap-trim-minclip",
                    scale=2,
                    min_width=160,
                )
                controls[AUTO_EDITOR_WORKFLOW_EXPORT_KEY] = gr.Dropdown(
                    choices=AUTO_EDITOR_WORKFLOW_EXPORT_CHOICES,
                    value=AUTO_EDITOR_WORKFLOW_EXPORT_NONE,
                    label="Auto-Editor workflow export",
                    info=(
                        "For Process File only. Exports an Auto-Editor timeline or "
                        "editor project instead of rendering media, using the current "
                        "trim settings so you can inspect or edit the planned cuts."
                    ),
                    elem_id="acestep-ap-auto-editor-workflow-export",
                    scale=2,
                    min_width=190,
                )
            add_video_auto_editor_controls(controls)

        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            add_single_file_controls(controls)

        add_stage_controls(controls)

        add_diffpitcher_controls(controls)

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
        gr.Markdown(
            "### Vocal Processing Order\n"
            "Source song -> SAM Audio / ACE-Step Extract vocals -> optional light cleanup "
            "only: trim silence, remove obvious noise, normalize safely -> DiffPitcher "
            "pitch fix -> Audio Enhancement / mastering -> final export"
        )
    return controls
