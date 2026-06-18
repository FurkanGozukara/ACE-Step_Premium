"""Custom/Remix strength, Retake, and Edit controls for the generation tab.

The top row keeps mode-level controls together: strength, mode help, Retake,
Edit, and Remix-only no_fsq/help. Expanded Retake/Edit detail panels remain
directly underneath. The ``Copy from current`` button click handler is wired in
``generation_run_wiring.py``.
"""

from typing import Any

import gradio as gr

from acestep.prompt_wildcards import WILDCARD_HELP_MARKDOWN
from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.events.generation.strength_defaults import (
    DEFAULT_AUDIO_COVER_STRENGTH,
)
from acestep.ui.gradio.i18n import t


def build_variation_morph_controls() -> dict[str, Any]:
    """Build strength, help, Retake, and Edit controls with expandable panels."""

    with gr.Row(equal_height=True) as strength_variation_row:
        with gr.Column(scale=5, min_width=320):
            audio_cover_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=DEFAULT_AUDIO_COVER_STRENGTH,
                step=0.01,
                label=t("generation.codes_strength_label"),
                info=t("generation.codes_strength_info"),
                elem_classes=["has-info-container"],
                visible=True,
            )
        with gr.Column(scale=1, min_width=120) as custom_help_group:
            gr.Markdown(t("generation.custom_help_label"))
            create_help_button("generation_custom")
        with gr.Column(scale=1, min_width=150) as variation_group:
            with gr.Row():
                retake_enabled = gr.Checkbox(
                    label="Retake",
                    value=False,
                    scale=8,
                    info="Create a controlled variation from the same seed and settings.",
                    elem_classes=["has-info-container"],
                )
                create_help_button("generation_retake")
        with gr.Column(scale=1, min_width=150) as flow_edit_column:
            with gr.Row():
                flow_edit_morph = gr.Checkbox(
                    label="Edit",
                    value=False,
                    scale=8,
                    info=(
                        "Whole-track prompt edit for Source Audio. Use it "
                        "to change style or lyrics across the uploaded "
                        "audio while keeping its timing/arrangement as much "
                        "as possible. Different from Repaint: Repaint "
                        "replaces only a selected start/end time range."
                    ),
                    elem_classes=["has-info-container"],
                )
                create_help_button("generation_edit")
        with gr.Column(scale=2, min_width=220, visible=False) as no_fsq_column:
            no_fsq = gr.Checkbox(
                label=t("generation.remix_no_fsq_label"),
                value=False,
                info=t("generation.remix_no_fsq_info"),
            )
        with gr.Column(scale=1, min_width=120, visible=False) as remix_help_group:
            gr.Markdown(t("generation.remix_help_label"))
            create_help_button("generation_remix")

    with gr.Row(equal_height=False):
        # ---- LEFT column: Retake details ----
        with gr.Column(scale=1, min_width=200):
            with gr.Group(visible=False) as retake_panel:
                with gr.Row():
                    retake_variance = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                        label="variance", scale=2,
                        info="0=baseline; 0.05-0.15 subtle; 0.5+ strong.",
                    )
                    retake_seed = gr.Textbox(
                        label="seed", value="", scale=1,
                        placeholder="empty=random",
                        info="Optional independent seed for Retake noise.",
                    )
                retake_think_warning = gr.Markdown(
                    "**Think is on: Retake will mix LM drift with "
                    "noise drift.** To retake a Think-mode result "
                    "cleanly: open the result's Score & LRC & LM "
                    "Codes panel, copy its **LM Codes** into the "
                    "**LM Codes Hints** field above, then uncheck Think "
                    "before adjusting variance. See the (?) help for "
                    "the full workflow.",
                    visible=False,
                )
        # ---- RIGHT column: Edit details ----
        with gr.Column(scale=1, min_width=200):
            with gr.Group(visible=False) as morph_panel:
                with gr.Row():
                    flow_edit_copy_from_current_btn = gr.Button(
                        "Copy current -> source",
                        variant="secondary", size="lg", scale=0, min_width=180,
                        elem_classes=["action-btn", "action-btn-preview"],
                    )
                gr.Markdown(
                    "**Edit changes the whole uploaded Source Audio using "
                    "two prompts.** The source fields below describe the "
                    "original audio. The top **Music Caption / Lyrics** "
                    "describe the target result. Use **Copy current -> "
                    "source** before rewriting the top fields.\n\n"
                    "**Custom + Edit:** turns Source Audio into the starting "
                    "point instead of generating from silence. Good for "
                    "changing an existing track's style or lyrics.\n\n"
                    "**Remix + Edit:** keeps Remix's source-audio scaffold, "
                    "but applies the difference between the source prompt "
                    "and target prompt more directly. Good for smoother "
                    "style/lyric changes than a normal Remix.\n\n"
                    "**Repaint:** does not use this Edit path. Repaint is "
                    "for replacing one selected time range while preserving "
                    "the rest of the audio."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        flow_edit_source_caption = gr.Textbox(
                            label="source caption",
                            placeholder="Describe the ORIGINAL audio.",
                            lines=4, max_lines=8,
                            info=(
                                "Original/source description. This should match "
                                "the uploaded Source Audio before the edit."
                            ),
                        )
                        gr.Markdown(
                            WILDCARD_HELP_MARKDOWN,
                            elem_classes=["ace-wildcard-help"],
                        )
                    with gr.Column(scale=1):
                        flow_edit_source_lyrics = gr.Textbox(
                            label="source lyrics",
                            placeholder="Original lyrics; top-level lyrics is the target.",
                            lines=4, max_lines=8,
                            info=(
                                "Original/source lyrics. The top Lyrics field is "
                                "the target lyrics after the edit."
                            ),
                        )
                        gr.Markdown(
                            WILDCARD_HELP_MARKDOWN,
                            elem_classes=["ace-wildcard-help"],
                        )
                with gr.Row():
                    flow_edit_n_min = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                        label="n_min",
                        info="Start of the diffusion window where edit force is applied.",
                    )
                    flow_edit_n_max = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                        label="n_max",
                        info=(
                            "End of the diffusion window; 1.0 applies "
                            "through the full schedule."
                        ),
                    )
                    flow_edit_n_avg = gr.Slider(
                        minimum=1, maximum=8, value=1, step=1,
                        label="n_avg",
                        info="Monte Carlo samples per step. Higher is steadier but slower.",
                    )
        # Visibility chains.
        retake_enabled.change(
            lambda v: gr.update(visible=bool(v)),
            inputs=[retake_enabled], outputs=[retake_panel],
        )
        flow_edit_morph.change(
            lambda v: gr.update(visible=bool(v)),
            inputs=[flow_edit_morph], outputs=[morph_panel],
        )
    return {
        "strength_variation_row": strength_variation_row,
        "audio_cover_strength": audio_cover_strength,
        "custom_help_group": custom_help_group,
        "variation_group": variation_group,
        "remix_help_group": remix_help_group,
        "no_fsq": no_fsq,
        "no_fsq_column": no_fsq_column,
        "retake_enabled": retake_enabled,
        "retake_panel": retake_panel,
        "retake_variance": retake_variance,
        "retake_seed": retake_seed,
        "retake_think_warning": retake_think_warning,
        "flow_edit_morph": flow_edit_morph,
        "flow_edit_column": flow_edit_column,
        "morph_panel": morph_panel,
        "flow_edit_copy_from_current_btn": flow_edit_copy_from_current_btn,
        "flow_edit_source_caption": flow_edit_source_caption,
        "flow_edit_source_lyrics": flow_edit_source_lyrics,
        "flow_edit_n_min": flow_edit_n_min,
        "flow_edit_n_max": flow_edit_n_max,
        "flow_edit_n_avg": flow_edit_n_avg,
    }
