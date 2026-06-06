"""DiffPitcher controls for the Audio Processing page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.interfaces.source_audio_preview import (
    AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
    AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
)


_REFERENCE_FILE_TYPES = [
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus",
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
]


def add_diffpitcher_controls(controls: dict[str, Any]) -> None:
    """Add DiffPitcher pitch-fix controls to Audio Processing."""

    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        gr.Markdown(
            "### DiffPitcher Pitch Fixer\n"
            "Useful when the vocal is isolated but sings the wrong notes. Give it the "
            "correct melody as a MIDI score or a guide vocal for the same song/phrase. "
            "It is not voice transfer and not a singer-style reference. Different-song "
            "references are skipped because they create broken pitch guides."
        )
        with gr.Row(equal_height=True):
            controls["ap_diffpitcher_enabled"] = gr.Checkbox(
                label="Enable DiffPitcher",
                value=False,
                info="Off by default. Turn on only for isolated vocal stems.",
                scale=1,
                min_width=180,
            )
            controls["ap_diffpitcher_mode"] = gr.Dropdown(
                choices=[
                    ("Template reference vocal", "template"),
                    ("MIDI score", "score"),
                ],
                value="template",
                label="Pitch Guide",
                scale=2,
                min_width=190,
            )
            with gr.Column(scale=3, min_width=280):
                gr.Markdown(
                    "**Template:** guide vocal singing this same melody. "
                    "**MIDI:** note file for this vocal line. "
                    "Not for copying another singer or another song."
                )
            controls["ap_diffpitcher_device"] = gr.Dropdown(
                choices=[("Auto", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
                value="auto",
                label="Device",
                info="Auto uses CUDA when available, otherwise CPU.",
                scale=1,
                min_width=140,
            )
        with gr.Row(equal_height=True):
            controls["ap_diffpitcher_reference_audio"] = gr.File(
                label="Reference Vocal Template (audio/video)",
                type="filepath",
                file_count="single",
                file_types=_REFERENCE_FILE_TYPES,
                key="audio_processing_diffpitcher_reference_upload",
                preserved_by_key=[],
                scale=2,
                min_width=240,
            )
            controls["ap_diffpitcher_midi"] = gr.File(
                label="MIDI Score (.mid, .midi)",
                type="filepath",
                file_count="single",
                file_types=[".mid", ".midi"],
                key="audio_processing_diffpitcher_midi_upload",
                preserved_by_key=[],
                scale=2,
                min_width=220,
            )
        with gr.Row():
            controls["ap_diffpitcher_reference_audio_preview"] = gr.Audio(
                label="Reference Vocal Audio",
                type="filepath",
                interactive=False,
                visible=False,
                elem_classes=AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES,
                waveform_options=AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS,
            )
            controls["ap_diffpitcher_reference_video_preview"] = gr.Video(
                label="Reference Vocal Video",
                interactive=False,
                visible=False,
                elem_classes=["ace-video-preview"],
            )
        controls["ap_diffpitcher_reference_status"] = gr.Markdown(
            "Select a reference vocal audio or video file for template mode.",
            elem_classes=["ace-status-scroll-10"],
        )
        with gr.Row(equal_height=True):
            controls["ap_diffpitcher_steps"] = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=50,
                label="Diffusion Steps",
                info="50 is the original inference default.",
                scale=2,
                min_width=180,
            )
            controls["ap_diffpitcher_shift_semitones"] = gr.Slider(
                minimum=-24.0,
                maximum=24.0,
                step=0.1,
                value=0.0,
                label="Pitch Shift (semitones)",
                info="Applies after the reference or MIDI pitch guide is built.",
                scale=2,
                min_width=220,
            )
            controls["ap_diffpitcher_mask_with_source"] = gr.Checkbox(
                label="Keep source voicing mask",
                value=True,
                info="For MIDI mode, keeps unvoiced source frames unvoiced.",
                scale=1,
                min_width=190,
            )
