"""LoRA synchronization wiring for the simple Generate Song tab."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.core.generation.handler.lora.folder_scan import lora_dropdown_choices

from .. import generation_handlers as gen_h


_LORA_SCALE_MAX = 3.0


def register_simple_lora_sync_handlers(
    *,
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
) -> None:
    """Mirror simple-tab LoRA controls into the advanced generation controls."""

    if "simple_lora_dropdown" not in simple_page or "simple_lora_scale_slider" not in simple_page:
        return

    simple_page["simple_lora_dropdown"].input(
        fn=sync_simple_lora_dropdown,
        inputs=[simple_page["simple_lora_dropdown"]],
        outputs=[
            generation_section["lora_dropdown"],
            generation_section["lora_path"],
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
        ],
    )
    simple_page["simple_lora_scale_slider"].input(
        fn=sync_simple_lora_scale,
        inputs=[
            simple_page["simple_lora_scale_slider"],
            generation_section["lora_path"],
            generation_section["lora_dropdown"],
        ],
        outputs=[
            generation_section["lora_scale_slider"],
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
        ],
    )
    generation_section["lora_dropdown"].input(
        fn=sync_advanced_lora_dropdown,
        inputs=[generation_section["lora_dropdown"]],
        outputs=[simple_page["simple_lora_dropdown"]],
    )
    generation_section["lora_scale_slider"].input(
        fn=sync_advanced_lora_scale,
        inputs=[generation_section["lora_scale_slider"]],
        outputs=[simple_page["simple_lora_scale_slider"]],
    )
    generation_section["refresh_lora_dropdown_btn"].click(
        fn=refresh_simple_lora_dropdown,
        inputs=[generation_section["lora_dropdown"]],
        outputs=[simple_page["simple_lora_dropdown"]],
    )


def sync_simple_lora_dropdown(selected_path: str | None) -> tuple[Any, Any, str, Any]:
    """Return updates that make the advanced LoRA controls match simple selection."""

    path_update, status, use_lora_update = gen_h.select_lora_dropdown_path(selected_path)
    return gr.update(value=selected_path or ""), path_update, status, use_lora_update


def sync_simple_lora_scale(
    scale: float | int | str | None,
    manual_path: str | None,
    selected_path: str | None,
) -> tuple[Any, str, Any]:
    """Return updates that mirror the simple LoRA scale into advanced controls."""

    status, use_lora_update = gen_h.update_lora_next_run_status(manual_path, selected_path)
    return gr.update(value=_normalize_lora_scale(scale)), status, use_lora_update


def sync_advanced_lora_dropdown(selected_path: str | None) -> Any:
    """Return an update that mirrors the advanced dropdown into the simple tab."""

    return gr.update(value=selected_path or "")


def sync_advanced_lora_scale(scale: float | int | str | None) -> Any:
    """Return an update that mirrors the advanced LoRA scale into the simple tab."""

    return gr.update(value=_normalize_lora_scale(scale))


def refresh_simple_lora_dropdown(current_value: str | None = None) -> Any:
    """Refresh simple-tab LoRA choices after the advanced refresh button is clicked."""

    choices = lora_dropdown_choices()
    valid_values = {value for _label, value in choices}
    value = current_value if current_value in valid_values else ""
    return gr.update(choices=choices, value=value)


def _normalize_lora_scale(scale: float | int | str | None) -> float:
    """Return a LoRA scale constrained to the shared UI slider range."""

    try:
        value = float(scale if scale is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(_LORA_SCALE_MAX, value))
