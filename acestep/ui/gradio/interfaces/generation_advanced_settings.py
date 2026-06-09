"""Top-level advanced settings builder for generation UI."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.events.generation.model_config import (
    get_ui_control_config_for_path,
    get_ui_control_config,
    select_preferred_model_path,
)
from acestep.ui.gradio.i18n import t

from .generation_advanced_dit_controls import build_dit_controls
from .generation_advanced_output_controls import (
    build_automation_controls,
    build_output_controls,
)
from .generation_advanced_primary_controls import (
    build_lm_controls,
    build_lora_controls,
)
from .generation_defaults import compute_init_defaults
from .generation_service_config import create_service_config_content


def create_advanced_settings_section(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
    show_compile_toggle: bool = True,
) -> dict[str, Any]:
    """Create the Settings accordion and return advanced generation controls.

    Args:
        dit_handler: DiT service handler used for model-aware control defaults.
        llm_handler: LM service handler used for LM/service configuration controls.
        init_params: Optional startup state used to prefill control values.
        language: UI language code.
        show_compile_toggle: Whether service settings render the compile checkbox visibly.

    Returns:
        A merged component map for service, LoRA, DiT, LM, output, and automation controls.
    """

    defaults = compute_init_defaults(init_params, language)
    service_pre_initialized = defaults["service_pre_initialized"]
    service_mode = defaults["service_mode"]
    lm_initialized = defaults["lm_initialized"]

    if service_pre_initialized and init_params and "dit_handler" in init_params:
        config_path = init_params.get("config_path", "")
        if init_params["dit_handler"].is_turbo_model():
            ui_config = get_ui_control_config(True)
        else:
            ui_config = get_ui_control_config_for_path(config_path)
    else:
        selected_config_path = (init_params or {}).get(
            "config_path"
        ) or select_preferred_model_path(
            dit_handler.get_available_acestep_v15_models(),
        )
        ui_config = get_ui_control_config_for_path(selected_config_path)

    with gr.Accordion(
        t("generation.advanced_settings"),
        open=True,
    ) as advanced_settings_accordion:
        service_components = create_service_config_content(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            defaults=defaults,
            init_params=init_params,
            show_compile_toggle=show_compile_toggle,
        )
        lora_components = build_lora_controls()
        dit_components = build_dit_controls(ui_config, think_enabled=lm_initialized)
        lm_components = build_lm_controls(
            service_mode=service_mode,
            generate_lm_audio_codes_default=True,
        )
        output_components = build_output_controls(
            service_pre_initialized=service_pre_initialized,
            service_mode=service_mode,
            init_params=init_params,
        )
        automation_components = build_automation_controls(service_mode=service_mode)

    result: dict[str, Any] = {"advanced_settings_accordion": advanced_settings_accordion}
    result.update(dit_components)
    result.update(lm_components)
    result.update(output_components)
    result.update(automation_components)
    result.update(lora_components)
    result.update(service_components)
    return result
