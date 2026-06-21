"""Primary advanced-settings controls for generation UI."""

from typing import Any

import gradio as gr

from acestep.core.generation.handler.lora.folder_scan import lora_dropdown_choices
from acestep.ui.gradio.i18n import t


def build_lora_controls() -> dict[str, Any]:
    """Create LoRA adapter controls for selecting and scaling inference adapters.

    Args:
        None.

    Returns:
        A component map containing LoRA selection, scale, and status controls.
    """

    with gr.Accordion(t("generation.lora_accordion_title"), open=True, elem_classes=["has-info-container"]):
        with gr.Row():
            lora_path = gr.Textbox(
                label=t("generation.lora_path_label"),
                placeholder=t("generation.lora_path_placeholder"),
                info=t("generation.lora_path_info"),
                scale=3,
            )
            lora_dropdown = gr.Dropdown(
                label=t("generation.lora_dropdown_label"),
                choices=lora_dropdown_choices(),
                value="",
                info=t("generation.lora_dropdown_info"),
                scale=2,
                interactive=True,
            )
            refresh_lora_dropdown_btn = gr.Button(
                t("generation.refresh_lora_dropdown_btn"),
                variant="primary",
                scale=2,
                size="lg",
                elem_classes=["action-btn", "action-btn-preview"],
            )
        with gr.Row():
            use_lora_checkbox = gr.Checkbox(
                value=False,
                visible=False,
            )
            lora_scale_slider = gr.Slider(
                minimum=0.0,
                maximum=3.0,
                value=1.0,
                step=0.05,
                label=t("generation.lora_scale_label"),
                info=t("generation.lora_scale_info"),
                scale=2,
            )
        lora_status = gr.Textbox(
            label=t("generation.lora_status_label"),
            value=t("generation.lora_status_default"),
            interactive=False,
            lines=1,
            elem_classes=["no-tooltip"],
        )
    return {
        "lora_path": lora_path,
        "lora_dropdown": lora_dropdown,
        "refresh_lora_dropdown_btn": refresh_lora_dropdown_btn,
        "use_lora_checkbox": use_lora_checkbox,
        "lora_scale_slider": lora_scale_slider,
        "lora_status": lora_status,
    }


def build_lm_controls(
    service_mode: bool,
    generate_lm_audio_codes_default: bool = True,
) -> dict[str, Any]:
    """Create language-model generation controls for advanced settings.

    Args:
        service_mode: Whether the UI is running in service mode (disables some controls).
        generate_lm_audio_codes_default: Initial semantic-code generation toggle value.

    Returns:
        A component map containing LM sampling, CoT, negative prompt, and batch controls.
    """

    with gr.Accordion(t("generation.advanced_lm_section"), open=True, elem_classes=["has-info-container"]):
        with gr.Row():
            lm_temperature = gr.Slider(
                label=t("generation.lm_temperature_label"),
                minimum=0.0,
                maximum=2.0,
                value=0.85,
                step=0.1,
                scale=1,
                info=t("generation.lm_temperature_info"),
                elem_classes=["has-info-container"],
            )
            lm_cfg_scale = gr.Slider(
                label=t("generation.lm_cfg_scale_label"),
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
                scale=1,
                info=t("generation.lm_cfg_scale_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            lm_top_k = gr.Slider(
                label=t("generation.lm_top_k_label"),
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                scale=1,
                info=t("generation.lm_top_k_info"),
                elem_classes=["has-info-container"],
            )
            lm_top_p = gr.Slider(
                label=t("generation.lm_top_p_label"),
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.01,
                scale=1,
                info=t("generation.lm_top_p_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            lm_negative_prompt = gr.Textbox(
                label=t("generation.lm_negative_prompt_label"),
                value="NO USER INPUT",
                placeholder=t("generation.lm_negative_prompt_placeholder"),
                info=t("generation.lm_negative_prompt_info"),
                elem_classes=["has-info-container"],
                lines=2,
            )
        with gr.Row():
            use_cot_metas = gr.Checkbox(
                label=t("generation.cot_metas_label"),
                value=True,
                info=t("generation.cot_metas_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )
            use_cot_language = gr.Checkbox(
                label=t("generation.cot_language_label"),
                value=False,
                info=t("generation.cot_language_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )
            constrained_decoding_debug = gr.Checkbox(
                label=t("generation.constrained_debug_label"),
                value=False,
                info=t("generation.constrained_debug_info"),
                scale=1,
                interactive=not service_mode,
            )
            lm_use_legacy_cfg_prompt = gr.Checkbox(
                label=t("generation.lm_use_legacy_cfg_prompt_label"),
                value=False,
                info=t("generation.lm_use_legacy_cfg_prompt_info"),
                scale=1,
                interactive=not service_mode,
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            allow_lm_batch = gr.Checkbox(
                label=t("generation.parallel_thinking_label"),
                value=False,
                info=t("generation.parallel_thinking_info"),
                scale=1,
                visible=False,
                elem_classes=["has-info-container"],
            )
            use_cot_caption = gr.Checkbox(
                label=t("generation.caption_rewrite_label"),
                value=False,
                info=t("generation.caption_rewrite_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )
            generate_lm_audio_codes = gr.Checkbox(
                label=t("generation.lm_audio_codes_label"),
                value=generate_lm_audio_codes_default,
                info=t("generation.lm_audio_codes_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )

    return {
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_language": use_cot_language,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch,
        "use_cot_caption": use_cot_caption,
        "generate_lm_audio_codes": generate_lm_audio_codes,
        "lm_use_legacy_cfg_prompt": lm_use_legacy_cfg_prompt,
    }
