"""Secondary generation-tab controls (cover, custom prompt, repaint)."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.interfaces.source_audio_preview import (
    GENERATION_REFERENCE_PREVIEW_ELEM_ID,
    GENERATION_UPLOAD_PREVIEW_ELEM_CLASSES,
    TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
)
from acestep.ui.gradio.media_file_types import MEDIA_FILE_TYPES
from acestep.ui.gradio.premium_features import (
    DEFAULT_PRESET_CAPTION,
    DEFAULT_PRESET_LYRICS,
)


def build_cover_strength_controls() -> dict[str, Any]:
    """Create code/remix strength controls used by non-simple generation modes.

    Args:
        None.

    Returns:
        A component map containing audio/code strength sliders and remix help group.
    """

    audio_cover_strength = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=1.0,
        step=0.01,
        label=t("generation.codes_strength_label"),
        info=t("generation.codes_strength_info"),
        elem_classes=["has-info-container"],
        visible=True,
    )
    with gr.Group(visible=False) as remix_help_group:
        create_help_button("generation_remix")
        no_fsq = gr.Checkbox(
            label="no_fsq",
            value=False,
            info=(
                "Use source-audio latents directly instead of FSQ-quantized "
                "codes. Try this when Remix loses detail; leave off for the "
                "standard cover path."
            ),
        )
    cover_noise_strength = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.0,
        step=0.01,
        label=t("generation.cover_noise_strength_label"),
        info=t("generation.cover_noise_strength_info"),
        elem_classes=["has-info-container"],
        visible=False,
    )
    return {
        "audio_cover_strength": audio_cover_strength,
        "remix_help_group": remix_help_group,
        "no_fsq": no_fsq,
        "cover_noise_strength": cover_noise_strength,
    }


def build_custom_mode_controls() -> dict[str, Any]:
    """Create custom-mode caption, lyrics, and reference-audio controls.

    Args:
        None.

    Returns:
        A component map containing custom-mode text/audio inputs and formatting actions.
    """

    with gr.Group(visible=True, elem_classes=["has-info-container"]) as custom_mode_group:
        create_help_button("generation_custom")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=200):
                reference_audio = gr.File(
                    label=t("generation.reference_audio"),
                    file_count="multiple",
                    type="filepath",
                    file_types=MEDIA_FILE_TYPES,
                    show_label=True,
                    elem_id="acestep-advanced-reference-audio-upload",
                    elem_classes=["has-info-container"],
                    key="advanced_reference_audio_upload",
                    preserved_by_key=[],
                )
                gr.Markdown(t("generation.reference_audio_info"))
                reference_audio_preview = gr.Audio(
                    label="Reference Audio Preview",
                    type="filepath",
                    sources=["upload"],
                    interactive=True,
                    editable=True,
                    visible=False,
                    elem_id=GENERATION_REFERENCE_PREVIEW_ELEM_ID,
                    elem_classes=GENERATION_UPLOAD_PREVIEW_ELEM_CLASSES,
                    waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
                )
                reference_video_preview = gr.Video(
                    label="Reference Video Preview",
                    interactive=False,
                    visible=False,
                    elem_classes=["ace-video-preview"],
                )
            with gr.Column(scale=8):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        captions = gr.Textbox(
                            label=t("generation.caption_label"),
                            placeholder=t("generation.caption_placeholder"),
                            value=DEFAULT_PRESET_CAPTION,
                            lines=12,
                            max_lines=12,
                            info=t("generation.caption_info"),
                            elem_classes=["has-info-container"],
                        )
                        with gr.Row(elem_classes="instrumental-row"):
                            format_caption_btn = gr.Button(
                                t("generation.format_caption_btn"),
                                variant="secondary",
                                size="lg",
                                elem_classes=["action-btn", "action-btn-preview"],
                            )
                    with gr.Column(scale=1):
                        lyrics = gr.Textbox(
                            label=t("generation.lyrics_label"),
                            placeholder=t("generation.lyrics_placeholder"),
                            value=DEFAULT_PRESET_LYRICS,
                            lines=18,
                            max_lines=18,
                            info=t("generation.lyrics_info"),
                            elem_classes=["has-info-container"],
                        )
                        with gr.Row(elem_classes="instrumental-row"):
                            instrumental_checkbox = gr.Checkbox(
                                label=t("generation.instrumental_label"),
                                value=False,
                                scale=1,
                                info=t("generation.instrumental_info"),
                                elem_classes=["has-info-container"],
                            )
                            format_lyrics_btn = gr.Button(
                                t("generation.format_lyrics_btn"),
                                variant="secondary",
                                size="lg",
                                scale=2,
                                elem_classes=["action-btn", "action-btn-clear"],
                            )
            with gr.Column(scale=1, min_width=80, elem_classes="icon-btn-wrap"):
                sample_btn = gr.Button(
                    t("generation.sample_btn"),
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-open"],
                )
    return {
        "custom_mode_group": custom_mode_group,
        "reference_audio": reference_audio,
        "reference_audio_preview": reference_audio_preview,
        "reference_video_preview": reference_video_preview,
        "captions": captions,
        "format_caption_btn": format_caption_btn,
        "lyrics": lyrics,
        "instrumental_checkbox": instrumental_checkbox,
        "format_lyrics_btn": format_lyrics_btn,
        "sample_btn": sample_btn,
    }


def build_repainting_controls() -> dict[str, Any]:
    """Create range controls used by repaint, lego, and complete flows.

    Args:
        None.

    Returns:
        A component map containing repainting group, header, and start/end controls.
    """

    with gr.Group(visible=False) as repainting_group:
        create_help_button("generation_repaint")
        repainting_header_html = gr.HTML(f"<h5>{t('generation.repainting_controls')}</h5>")
        with gr.Row():
            repainting_start = gr.Number(
                label=t("generation.repainting_start"),
                value=0.0,
                step=0.1,
                info=t("generation.repainting_start_info"),
                elem_classes=["has-info-container"],
            )
            repainting_end = gr.Number(
                label=t("generation.repainting_end"),
                value=-1,
                minimum=-1,
                step=0.1,
                info=t("generation.repainting_end_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            repaint_mode = gr.Dropdown(
                label="Repaint Mode",
                choices=["conservative", "balanced", "aggressive"],
                value="balanced",
                info=t("generation.repaint_mode_info"),
                elem_classes=["has-info-container"],
            )
            repaint_strength = gr.Slider(
                label="Repaint Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.5,
                info=t("generation.repaint_strength_info"),
                elem_classes=["has-info-container"],
            )
        repaint_strength_memory = gr.State(value=0.5)
    return {
        "repainting_group": repainting_group,
        "repainting_header_html": repainting_header_html,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "repaint_mode": repaint_mode,
        "repaint_strength": repaint_strength,
        "repaint_strength_memory": repaint_strength_memory,
    }


