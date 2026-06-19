"""Generation-tab section orchestrator for the Gradio interface."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.events.generation.model_config import (
    get_generation_mode_choices_for_path,
)

from .generation_defaults import compute_init_defaults, resolve_generation_config_path
from .generation_tab_primary_controls import (
    build_hidden_generation_state,
    build_mode_selector_controls,
)
from .generation_tab_simple_controls import (
    build_simple_mode_controls,
)
from .generation_tab_source_controls import (
    build_source_track_and_code_controls,
)
from .generation_tab_generate_controls import build_generate_row_controls
from .generation_tab_optional_controls import (
    build_optional_parameter_controls,
)
from .generation_tab_secondary_controls import (
    build_custom_mode_controls,
    build_repainting_controls,
)
from .generation_tab_variation_morph_controls import build_variation_morph_controls


def _compute_generation_tab_defaults(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Compute shared defaults used by both split and legacy generation layouts."""

    _ = llm_handler  # retained for caller signature parity
    defaults = compute_init_defaults(init_params, language)
    service_pre_initialized = defaults["service_pre_initialized"]

    config_path = resolve_generation_config_path(
        dit_handler=dit_handler,
        init_params=init_params,
        service_pre_initialized=service_pre_initialized,
    )
    initial_mode_choices = get_generation_mode_choices_for_path(config_path)

    return {
        "defaults": defaults,
        "initial_mode_choices": initial_mode_choices,
    }


def create_generation_mode_section(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Create only the top generation-mode controls for flexible page layouts."""

    generation_defaults = _compute_generation_tab_defaults(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        init_params=init_params,
        language=language,
    )
    return build_mode_selector_controls(generation_defaults["initial_mode_choices"])


def create_generation_body_section(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Create the main generation controls excluding the top mode selector row."""

    generation_defaults = _compute_generation_tab_defaults(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        init_params=init_params,
        language=language,
    )
    defaults = generation_defaults["defaults"]
    service_pre_initialized = defaults["service_pre_initialized"]
    service_mode = defaults["service_mode"]
    lm_initialized = defaults["lm_initialized"]
    max_duration = defaults["max_duration"]
    max_batch_size = defaults["max_batch_size"]
    default_batch_size = defaults["default_batch_size"]

    composition_guide = gr.Markdown(
        t("generation.composition_guide_custom"),
        elem_classes=["has-info-container"],
    )

    hidden_state_controls = build_hidden_generation_state()
    simple_mode_controls = build_simple_mode_controls()
    source_track_code_controls = build_source_track_and_code_controls(
        service_mode=service_mode,
        init_params=init_params,
    )
    variation_morph_controls = build_variation_morph_controls()
    custom_mode_controls = build_custom_mode_controls()
    repainting_controls = build_repainting_controls()
    optional_controls = build_optional_parameter_controls(
        max_duration=max_duration,
        max_batch_size=max_batch_size,
        default_batch_size=default_batch_size,
        service_mode=service_mode,
    )
    generate_controls = build_generate_row_controls(
        service_pre_initialized=service_pre_initialized,
        init_params=init_params,
        lm_initialized=lm_initialized,
        service_mode=service_mode,
    )

    result: dict[str, Any] = {}
    result.update(hidden_state_controls)
    result.update(simple_mode_controls)
    result.update(source_track_code_controls)
    result.update(variation_morph_controls)
    result.update(custom_mode_controls)
    result.update(repainting_controls)
    result.update(optional_controls)
    result.update(generate_controls)
    result.update(
        {
            "max_duration": max_duration,
            "max_batch_size": max_batch_size,
            "composition_guide": composition_guide,
        }
    )
    return result


def create_generation_tab_section(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Create generation-tab controls and mode-specific UI sections.

    Args:
        dit_handler: DiT service handler used for model-aware mode defaults.
        llm_handler: LM service handler retained for signature parity with callers.
        init_params: Optional startup state used to prefill runtime defaults.
        language: UI language code used for default computation.

    Returns:
        A merged component map for generation-tab controls and runtime metadata.
    """

    with gr.Group():
        mode_controls = create_generation_mode_section(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            init_params=init_params,
            language=language,
        )
        body_controls = create_generation_body_section(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            init_params=init_params,
            language=language,
        )

    result: dict[str, Any] = {}
    result.update(mode_controls)
    result.update(body_controls)
    return result
