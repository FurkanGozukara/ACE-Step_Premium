"""Generate-row controls for the generation tab."""

from typing import Any

import gradio as gr

from acestep.gpu_config import get_global_gpu_config
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.interfaces.source_audio_preview import (
    TRIM_AUDIO_PREVIEW_ELEM_CLASSES,
    TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
)
from .generation_tab_batch_extract_controls import build_batch_extract_controls


def _build_left_generate_toggles(
    lm_initialized: bool,
    service_mode: bool,
) -> tuple[gr.Checkbox, gr.Checkbox]:
    """Create the first two runtime toggles shown above the generate button.

    Args:
        lm_initialized: Whether LM is initialized.
        service_mode: Whether UI is in service mode, used to gate toggle interactivity.

    Returns:
        The ``think_checkbox`` and ``auto_score`` controls in creation order.
    """

    save_memory = get_global_gpu_config().save_memory_mode
    _ = lm_initialized
    think_checkbox = gr.Checkbox(
        label=t("generation.think_label"),
        value=True,
        visible=True,
        scale=1,
        interactive=not service_mode,
        info=t("generation.think_info"),
        elem_classes=["has-info-container"],
    )
    auto_score = gr.Checkbox(
        label=t("generation.auto_score_label"),
        value=False,
        scale=1,
        interactive=not service_mode and not save_memory,
        info=t("generation.auto_score_info"),
        elem_classes=["has-info-container"],
    )
    return think_checkbox, auto_score


def _build_right_generate_toggles(service_mode: bool) -> tuple[gr.Checkbox, gr.Checkbox]:
    """Create the last two runtime toggles shown above the generate button.

    Args:
        service_mode: Whether UI is in service mode, used to gate toggle interactivity.

    Returns:
        The ``autogen_checkbox`` and ``auto_lrc`` controls in creation order.
    """

    save_memory = get_global_gpu_config().save_memory_mode
    autogen_checkbox = gr.Checkbox(
        label=t("generation.autogen_label"),
        value=False,
        scale=1,
        interactive=not service_mode,
        info=t("generation.autogen_info"),
        elem_classes=["has-info-container"],
    )
    auto_lrc = gr.Checkbox(
        label=t("generation.auto_lrc_label"),
        value=False,
        scale=1,
        interactive=not service_mode and not save_memory,
        info=t("generation.auto_lrc_info"),
        elem_classes=["has-info-container"],
    )
    return autogen_checkbox, auto_lrc


def build_generate_row_controls(
    service_pre_initialized: bool,
    init_params: dict[str, Any] | None,
    lm_initialized: bool,
    service_mode: bool,
) -> dict[str, Any]:
    """Compose generate-row controls from toggle helpers and the generate button.

    Args:
        service_pre_initialized: Whether startup params should prefill interactive defaults.
        init_params: Optional startup state containing generate-button enable state.
        lm_initialized: Whether LM is initialized, used to gate think-checkbox interactivity.
        service_mode: Whether the UI is running in service mode (disables some controls).

    Returns:
        A component map containing generate button, runtime toggles, and container.
    """

    params = init_params or {}
    generate_btn_interactive = (
        params.get("enable_generate", True) if service_pre_initialized else True
    )
    with gr.Column(visible=True) as generate_btn_row:
        with gr.Row(equal_height=True) as runtime_options_row:
            think_checkbox, auto_score = _build_left_generate_toggles(
                lm_initialized=lm_initialized,
                service_mode=service_mode,
            )
            autogen_checkbox, auto_lrc = _build_right_generate_toggles(service_mode=service_mode)
        with gr.Row(equal_height=False, elem_classes=["ace-generate-action-row"]):
            generate_btn = gr.Button(
                t("generation.generate_btn"),
                variant="primary",
                size="lg",
                scale=2,
                interactive=generate_btn_interactive,
                elem_id="acestep-generate-btn",
                elem_classes=["action-btn", "action-btn-generate"],
            )
            cancel_generation_btn = gr.Button(
                "Cancel Generation",
                variant="stop",
                size="lg",
                scale=1,
                elem_classes=[
                    "action-btn",
                    "action-btn-cancel",
                    "action-btn-cancel-advanced",
                ],
            )
            generate_row_open_outputs_folder_btn = gr.Button(
                "Open Outputs Folder",
                variant="secondary",
                size="lg",
                scale=1,
                elem_classes=["action-btn", "action-btn-open"],
            )
        cancel_confirmed_state = gr.State(value=False)
        with gr.Group():
            gr.Markdown(t("generation.inline_result_info"))
            with gr.Row():
                inline_generated_audio = gr.Audio(
                    label=t("generation.inline_result_audio_label"),
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                    elem_classes=TRIM_AUDIO_PREVIEW_ELEM_CLASSES,
                    waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
                )
                inline_remaining_audio = gr.Audio(
                    label="Remaining Audio",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                    visible=False,
                    elem_classes=TRIM_AUDIO_PREVIEW_ELEM_CLASSES,
                    waveform_options=TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS,
                )
            inline_generation_status = gr.Textbox(
                label=t("generation.inline_result_status_label"),
                interactive=False,
                lines=3,
                max_lines=3,
                autoscroll=True,
            )
            batch_extract_controls = build_batch_extract_controls()
    return {
        "think_checkbox": think_checkbox,
        "auto_score": auto_score,
        "runtime_options_row": runtime_options_row,
        "generate_btn": generate_btn,
        "cancel_generation_btn": cancel_generation_btn,
        "generate_row_open_outputs_folder_btn": generate_row_open_outputs_folder_btn,
        "cancel_confirmed_state": cancel_confirmed_state,
        "generate_btn_row": generate_btn_row,
        "autogen_checkbox": autogen_checkbox,
        "auto_lrc": auto_lrc,
        "inline_generated_audio": inline_generated_audio,
        "inline_remaining_audio": inline_remaining_audio,
        "inline_generation_status": inline_generation_status,
        **batch_extract_controls,
    }
