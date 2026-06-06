"""Video Auto-Editor controls for the Audio Processing page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.video_reencode_settings import (
    DEFAULT_VIDEO_AUDIO_BITRATE,
    DEFAULT_VIDEO_AUDIO_CODEC,
    DEFAULT_VIDEO_AUTO_QUALITY,
    DEFAULT_VIDEO_BITRATE,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_PRESET,
    VIDEO_ENCODER_PRESET_CHOICES,
)


def add_video_auto_editor_controls(controls: dict[str, Any]) -> None:
    """Add Auto-Editor video reencode controls to the Audio Processing page."""

    with gr.Accordion("Auto-Editor Video Reencode", open=False):
        controls["ap_video_auto_quality"] = gr.Checkbox(
            label="Auto set quality",
            value=DEFAULT_VIDEO_AUTO_QUALITY,
            info="Matches source video/audio bitrates when Auto-Editor must render video.",
        )
        with gr.Row(equal_height=True):
            controls["ap_video_codec"] = gr.Textbox(
                label="Video codec",
                value=DEFAULT_VIDEO_CODEC,
                info="Used when Auto set quality is off.",
                scale=2,
            )
            controls["ap_video_bitrate"] = gr.Textbox(
                label="Video bitrate",
                value=DEFAULT_VIDEO_BITRATE,
                placeholder="Example: 6000k",
                info="Manual bitrate. Blank uses CRF.",
                scale=2,
            )
            controls["ap_video_crf"] = gr.Slider(
                minimum=0,
                maximum=63,
                step=1,
                value=DEFAULT_VIDEO_CRF,
                label="CRF",
                info="Lower is higher quality. Used when manual bitrate is blank.",
                scale=2,
            )
        with gr.Row(equal_height=True):
            controls["ap_video_preset"] = gr.Dropdown(
                choices=VIDEO_ENCODER_PRESET_CHOICES,
                value=DEFAULT_VIDEO_PRESET,
                label="Encoder preset",
                scale=2,
            )
            controls["ap_video_audio_codec"] = gr.Textbox(
                label="Audio codec",
                value=DEFAULT_VIDEO_AUDIO_CODEC,
                scale=2,
            )
            controls["ap_video_audio_bitrate"] = gr.Textbox(
                label="Audio bitrate",
                value=DEFAULT_VIDEO_AUDIO_BITRATE,
                placeholder="Example: 192k",
                scale=2,
            )
