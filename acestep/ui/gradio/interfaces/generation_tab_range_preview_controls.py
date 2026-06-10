"""Range-preview controls for the generation start/end row."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.interfaces.source_audio_preview import (
    GENERATION_UPLOAD_PREVIEW_ELEM_CLASSES,
    TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
)


def build_repainting_range_preview_controls() -> dict[str, Any]:
    """Create the media preview column for selected source start/end ranges.

    Returns:
        Component map containing audio and video previews for the selected range.
    """

    with gr.Column(scale=3, min_width=260):
        repainting_range_preview_audio = gr.Audio(
            label="Preview",
            type="filepath",
            interactive=False,
            visible=False,
            elem_id="acestep-repainting-range-preview-audio",
            elem_classes=[*GENERATION_UPLOAD_PREVIEW_ELEM_CLASSES, "ace-range-preview"],
            waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
        )
        repainting_range_preview_video = gr.Video(
            label="Preview",
            interactive=False,
            visible=False,
            elem_id="acestep-repainting-range-preview-video",
            elem_classes=["ace-video-preview", "ace-range-preview"],
        )
    return {
        "repainting_range_preview_audio": repainting_range_preview_audio,
        "repainting_range_preview_video": repainting_range_preview_video,
    }
