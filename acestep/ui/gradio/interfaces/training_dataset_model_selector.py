"""Dataset-builder model selection controls."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.events.generation.model_config import select_preferred_model_path


def _available_dit_models(dit_handler: Any) -> list[str]:
    """Return selectable ACE-Step DiT models for dataset actions."""

    if dit_handler is None:
        return []
    try:
        return list(dit_handler.get_available_acestep_v15_models())
    except Exception:
        return []


def _selected_dataset_model(
    available_models: list[str],
    init_params: dict[str, Any] | None,
    dit_handler: Any,
) -> str:
    """Return the model selected by default for dataset actions."""

    params = init_params if isinstance(init_params, dict) else {}
    initialized_params = getattr(dit_handler, "last_init_params", None) or {}
    candidates = (
        params.get("config_path"),
        initialized_params.get("config_path")
        if isinstance(initialized_params, dict)
        else None,
        select_preferred_model_path(available_models),
    )
    for candidate in candidates:
        model = str(candidate or "").strip()
        if model and (not available_models or model in available_models):
            return model
    return select_preferred_model_path(available_models)


def _build_model_selector(
    *,
    control_key: str,
    label: str,
    info: str,
    dit_handler: Any,
    init_params: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Render a DiT model selector for training workflows."""

    available_models = _available_dit_models(dit_handler)
    model_config = gr.Dropdown(
        label=label,
        choices=available_models,
        value=_selected_dataset_model(available_models, init_params, dit_handler),
        allow_custom_value=True,
        info=info,
        elem_classes=["has-info-container"],
    )
    return {control_key: model_config}


def build_dataset_model_selector(
    dit_handler: Any,
    init_params: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Render the DiT model selector used by dataset auto-label/preprocess."""

    return _build_model_selector(
        control_key="dataset_model_config",
        label="Dataset Model",
        dit_handler=dit_handler,
        init_params=init_params,
        info="DiT model used when dataset actions need to initialize the service.",
    )


def build_lora_training_model_selector(
    dit_handler: Any,
    init_params: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Render the DiT model selector used by LoRA training."""

    return _build_model_selector(
        control_key="lora_model_config",
        label="LoRA Base Model",
        dit_handler=dit_handler,
        init_params=init_params,
        info="DiT base model to load before LoRA training starts.",
    )
