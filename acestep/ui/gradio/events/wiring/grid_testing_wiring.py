"""Event wiring for Grid Testing."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.core.generation.handler.lora.folder_scan import lora_dropdown_choices
from acestep.ui.gradio.events.generation.cancel_actions import (
    BATCH_CANCEL_CONFIRM_JS,
    request_generation_cancel_from_ui,
)
from acestep.ui.gradio.events.grid_testing_loras import filter_grid_lora_choices
from acestep.ui.gradio.events.grid_testing_runner import run_grid_testing
from acestep.ui.gradio.events.local_path_dialogs import select_folder_path
from acestep.ui.gradio.events.wiring.generation_run_wiring import build_generation_run_inputs
from acestep.ui.gradio.events.wiring.simple_create_wiring import (
    build_simple_prepare_inputs,
    build_simple_prepare_outputs,
)
from acestep.ui.gradio.events.wiring.simple_create_params import prepare_simple_generation


def register_grid_testing_handlers(
    *,
    dit_handler: Any,
    llm_handler: Any,
    grid_section: dict[str, Any],
    simple_page: dict[str, Any],
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
) -> None:
    """Register Grid Testing button and picker handlers."""

    def grid_wrapper(selected_loras, output_folder, mp3_only, generations_per_lora, *args):
        """Stream Grid Testing status while processing selected LoRAs."""

        yield from run_grid_testing(
            dit_handler,
            llm_handler,
            selected_loras,
            output_folder,
            mp3_only,
            args,
            generations_per_lora=generations_per_lora,
        )

    grid_section["grid_lora_filter"].input(
        fn=_filter_grid_loras,
        inputs=[
            grid_section["grid_lora_filter"],
            grid_section["grid_lora_dropdown"],
        ],
        outputs=[grid_section["grid_lora_dropdown"]],
    )
    grid_section["grid_refresh_loras_btn"].click(
        fn=_refresh_grid_loras,
        inputs=[
            grid_section["grid_lora_filter"],
            grid_section["grid_lora_dropdown"],
        ],
        outputs=[grid_section["grid_lora_dropdown"]],
    )
    grid_section["grid_output_folder_browse_btn"].click(
        fn=select_folder_path,
        inputs=[grid_section["grid_output_folder"]],
        outputs=[grid_section["grid_output_folder"]],
    )
    grid_section["grid_generate_btn"].click(
        fn=prepare_simple_generation,
        inputs=build_simple_prepare_inputs(simple_page),
        outputs=build_simple_prepare_outputs(
            generation_section,
            results_section,
            grid_section["grid_status"],
        ),
    ).then(
        fn=grid_wrapper,
        inputs=[
            grid_section["grid_lora_dropdown"],
            grid_section["grid_output_folder"],
            grid_section["grid_mp3_only"],
            grid_section["grid_generation_count"],
            *build_generation_run_inputs(generation_section, results_section),
        ],
        outputs=[
            grid_section["grid_status"],
            grid_section["grid_generated_files"],
        ],
    )
    cancel_event = grid_section["grid_cancel_btn"].click(
        fn=None,
        inputs=None,
        outputs=[grid_section["grid_cancel_confirmed_state"]],
        js=BATCH_CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    cancel_event.then(
        fn=request_generation_cancel_from_ui,
        inputs=[
            grid_section["grid_cancel_confirmed_state"],
            generation_section["subprocess_mode_checkbox"],
        ],
        outputs=[grid_section["grid_status"]],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )


def _filter_grid_loras(filter_text: Any = "", current_values: Any = None) -> Any:
    """Filter Grid Testing LoRA choices while preserving current selections."""

    return _grid_lora_dropdown_update(filter_text, current_values)


def _refresh_grid_loras(filter_text: Any = "", current_values: Any = None) -> Any:
    """Refresh Grid Testing LoRA choices while preserving valid selections."""

    return _grid_lora_dropdown_update(filter_text, current_values)


def _grid_lora_dropdown_update(filter_text: Any = "", current_values: Any = None) -> Any:
    """Return a filtered LoRA dropdown update for Grid Testing."""

    choices = lora_dropdown_choices()
    visible_choices, selected = filter_grid_lora_choices(
        choices,
        filter_text=filter_text,
        selected_loras=current_values,
    )
    return gr.update(choices=visible_choices, value=selected)
