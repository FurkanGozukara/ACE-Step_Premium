"""UI helpers for browsing project LoRA folders."""

from __future__ import annotations

import gradio as gr

from acestep.core.generation.handler.lora.folder_scan import (
    discover_lora_folder_items,
    lora_dropdown_choices,
    resolve_loadable_lora_adapter_path,
)


def initial_lora_dropdown_choices() -> list[tuple[str, str]]:
    """Return initial choices and create the preferred ``Loras`` folder."""

    return lora_dropdown_choices()


def _effective_lora_path(
    manual_path: str | None = None,
    selected_path: str | None = None,
) -> str:
    """Return manual path if present, otherwise the dropdown selection."""

    manual = str(manual_path or "").strip()
    if manual:
        return manual
    return str(selected_path or "").strip()


def _next_run_status(path: str | None) -> tuple[str, bool]:
    """Return the user-facing next-run LoRA status and implicit use flag."""

    candidate = str(path or "").strip()
    if not candidate:
        return "No LoRA will be used.", False
    resolved = resolve_loadable_lora_adapter_path(candidate)
    if not resolved:
        return f"No LoRA will be used. Invalid LoRA path: {candidate}", False
    return f"Next run will use LoRA: {resolved}", True


def refresh_lora_dropdown(
    current_value: str | None = None,
    manual_path: str | None = None,
) -> tuple[dict, str, dict]:
    """Refresh the discovered LoRA dropdown choices."""

    choices = lora_dropdown_choices()
    values = {value for _label, value in choices}
    value = current_value if current_value in values else ""
    status, use_lora = _next_run_status(_effective_lora_path(manual_path, value))
    return gr.update(choices=choices, value=value), status, gr.update(value=use_lora)


def select_lora_dropdown_path(selected_path: str | None) -> tuple[dict, str, dict]:
    """Copy a dropdown LoRA selection into the manual path field."""

    selected = str(selected_path or "").strip()
    status, use_lora = _next_run_status(selected)
    return gr.update(value=selected), status, gr.update(value=use_lora)


def update_lora_next_run_status(
    manual_path: str | None = None,
    selected_path: str | None = None,
) -> tuple[str, dict]:
    """Describe whether the current LoRA path/dropdown will affect generation."""

    status, use_lora = _next_run_status(_effective_lora_path(manual_path, selected_path))
    return status, gr.update(value=use_lora)


def describe_lora_folder_scan() -> str:
    """Return a short scan summary for real smoke tests and diagnostics."""

    items = discover_lora_folder_items()
    return f"{len(items)} LoRA adapter(s): " + ", ".join(item.label for item in items[:20])
