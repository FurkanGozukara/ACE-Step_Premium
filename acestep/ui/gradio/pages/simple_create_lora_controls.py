"""Model and LoRA controls for the simple Generate Song tab."""

from __future__ import annotations

from typing import Any, Sequence

import gradio as gr

from acestep.core.generation.handler.lora.folder_scan import lora_dropdown_choices
from acestep.ui.gradio.i18n import t


_MODEL_INFO = (
    "SFT uses 50-step CFG with Thinking metadata and shift 3.0. Base uses 64-step "
    "APG/CFG with shift 3.0. Turbo uses 8-step fast defaults with shift 3.0. "
    "All are XL 4B models; >=12GB VRAM is the practical floor."
)
_LORA_SCALE_MAX = 3.0


def build_simple_model_lora_controls(
    *,
    default_model: str,
    model_choices: Sequence[tuple[str, str]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Create simple-tab model controls and grouped LoRA controls.

    Args:
        default_model: Initial model value for the simple model selector.
        model_choices: Model dropdown choices displayed in the simple tab.
        params: Startup parameter values used to prefill LoRA controls.

    Returns:
        Component map for the simple model, LoRA dropdown, and LoRA scale slider.
    """

    lora_choices = lora_dropdown_choices()
    lora_value = _selected_lora_value(params, lora_choices)

    with gr.Row():
        simple_model_dropdown = gr.Dropdown(
            choices=list(model_choices),
            value=default_model,
            label="Model",
            info=_MODEL_INFO,
            scale=3,
        )
        simple_sampler_mode = gr.Dropdown(
            choices=["euler", "heun"],
            value=_selected_sampler_mode(params),
            label=t("generation.sampler_mode_label"),
            info=t("generation.simple_sampler_mode_info"),
            scale=1,
        )

    with gr.Group():
        with gr.Row():
            simple_lora_dropdown = gr.Dropdown(
                label="Select LoRA",
                choices=lora_choices,
                value=lora_value,
                info=t("generation.lora_dropdown_info"),
                interactive=True,
                scale=4,
            )
            simple_refresh_lora_dropdown_btn = gr.Button(
                t("generation.refresh_lora_dropdown_btn"),
                variant="secondary",
                size="sm",
                scale=1,
                min_width=90,
            )
        simple_lora_scale_slider = gr.Slider(
            minimum=0.0,
            maximum=_LORA_SCALE_MAX,
            value=_selected_lora_scale(params),
            step=0.05,
            label=t("generation.lora_scale_label"),
            info=t("generation.lora_scale_info"),
        )

    return {
        "simple_model_dropdown": simple_model_dropdown,
        "simple_sampler_mode": simple_sampler_mode,
        "simple_lora_dropdown": simple_lora_dropdown,
        "simple_refresh_lora_dropdown_btn": simple_refresh_lora_dropdown_btn,
        "simple_lora_scale_slider": simple_lora_scale_slider,
    }


def _selected_lora_value(
    params: dict[str, Any],
    choices: Sequence[tuple[str, str]],
) -> str:
    """Return a simple-tab LoRA value that is present in the dropdown choices."""

    candidate = str(
        params.get("simple_lora_dropdown") or params.get("lora_dropdown") or ""
    ).strip()
    valid_values = {value for _label, value in choices}
    return candidate if candidate in valid_values else ""


def _selected_lora_scale(params: dict[str, Any]) -> float:
    """Return a valid simple-tab LoRA scale, defaulting to full strength."""

    raw_value = params.get("simple_lora_scale_slider", params.get("lora_scale_slider", 1.0))
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(_LORA_SCALE_MAX, value))


def _selected_sampler_mode(params: dict[str, Any]) -> str:
    """Return the shared sampler mode selected for Generate Song startup."""

    raw_value = params.get("simple_create_sampler_mode", params.get("sampler_mode", "heun"))
    value = str(raw_value or "").strip().lower()
    return value if value in {"euler", "heun"} else "heun"
