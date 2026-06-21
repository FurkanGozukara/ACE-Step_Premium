"""Helper functions for generation mode UI update construction."""

import gradio as gr

from acestep.ui.gradio.i18n import t


def _compute_field_updates_for_mode(
    is_extract: bool,
    is_lego: bool,
    not_simple: bool,
    leaving_extract_or_lego: bool,
):
    """Compute gr.update() for captions, lyrics, bpm, and key_scale."""
    if is_extract:
        return (
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=None, interactive=False, visible=False),
            gr.update(value="", interactive=False, visible=False),
        )
    if is_lego:
        return (
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            gr.update(value=None, interactive=False, visible=False),
            gr.update(value="", interactive=False, visible=False),
        )
    if not_simple:
        if leaving_extract_or_lego:
            return (
                gr.update(value="", visible=True, interactive=True),
                gr.update(value="", visible=True, interactive=True),
                gr.update(value=None, visible=True, interactive=False),
                gr.update(value="", visible=True, interactive=False),
            )
        return (
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    if leaving_extract_or_lego:
        return (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=""),
        )
    return gr.update(), gr.update(), gr.update(), gr.update()


def _compute_meta_updates_for_mode(
    is_extract: bool,
    is_lego: bool,
    not_simple: bool,
    leaving_extract_or_lego: bool,
):
    """Compute gr.update() for time_signature, vocal_language, audio_duration."""
    if is_extract:
        return (
            gr.update(value="", interactive=False, visible=False),
            gr.update(value="unknown", interactive=False, visible=False),
            gr.update(value=-1, interactive=False, visible=False),
        )
    if is_lego:
        return (
            gr.update(value="", interactive=False, visible=False),
            gr.update(visible=True, interactive=True),
            gr.update(value=-1, interactive=False, visible=False),
        )
    if not_simple:
        if leaving_extract_or_lego:
            return (
                gr.update(value="", visible=True, interactive=False),
                gr.update(visible=True, interactive=True),
                gr.update(value=-1, visible=True, interactive=False),
            )
        return (
            gr.update(visible=True),
            gr.update(visible=True, interactive=True),
            gr.update(visible=True),
        )
    if leaving_extract_or_lego:
        return (
            gr.update(value=""),
            gr.update(),
            gr.update(value=-1),
        )
    return gr.update(), gr.update(), gr.update()


def _compute_automation_updates(is_extract: bool, is_lego: bool, not_simple: bool):
    """Compute gr.update() for automation controls."""
    if is_extract or is_lego:
        return (
            gr.update(visible=False, value=False, interactive=False),
            gr.update(visible=False, value=False, interactive=False),
            gr.update(visible=False, value=False, interactive=False),
            gr.skip(),
        )
    if not_simple:
        return (
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            gr.skip(),
        )
    return gr.update(), gr.update(), gr.update(), gr.skip()


def _compute_repainting_labels(
    is_lego: bool,
    is_repaint: bool,
    is_complete: bool,
    is_cover: bool,
):
    """Compute gr.update() for repainting header, start, and end labels."""
    if is_cover:
        return _range_label_updates(
            "generation.remix_source_range_controls",
            "generation.remix_source_start",
            "generation.remix_source_end",
            "generation.remix_source_start_info",
            "generation.remix_source_end_info",
        )
    if is_lego:
        return _range_label_updates(
            "generation.stem_area_controls",
            "generation.stem_start",
            "generation.stem_end",
            "generation.stem_start_info",
            "generation.stem_end_info",
        )
    if is_complete:
        return _range_label_updates(
            "generation.complete_section_controls",
            "generation.complete_start",
            "generation.complete_end",
            "generation.complete_start_info",
            "generation.complete_end_info",
        )
    if is_repaint:
        return _range_label_updates(
            "generation.repainting_controls",
            "generation.repainting_start",
            "generation.repainting_end",
            "generation.repainting_start_info",
            "generation.repainting_end_info",
        )
    return gr.update(), gr.update(), gr.update()


def _range_label_updates(
    header_key: str,
    start_key: str,
    end_key: str,
    start_info_key: str,
    end_info_key: str,
):
    """Build updates for a mode-specific source range row."""

    return (
        gr.update(value=f"<h5>{t(header_key)}</h5>"),
        gr.update(label=t(start_key), info=t(start_info_key)),
        gr.update(label=t(end_key), info=t(end_info_key)),
    )

