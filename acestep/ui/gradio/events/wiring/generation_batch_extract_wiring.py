"""Event wiring for Advanced-tab Batch Extract."""

from __future__ import annotations

from typing import Any

from acestep.ui.gradio.events.batch_extract_runner import run_batch_extract_processing
from acestep.ui.gradio.events.generation.cancel_actions import (
    BATCH_CANCEL_CONFIRM_JS,
    request_full_generation_cancel_from_ui,
)
from acestep.ui.gradio.events.local_path_dialogs import select_folder_path
from acestep.ui.gradio.events.wiring.context import GenerationWiringContext
from acestep.ui.gradio.events.wiring.generation_run_wiring import build_generation_run_inputs


def register_generation_batch_extract_handlers(context: GenerationWiringContext) -> None:
    """Register folder Batch Extract controls on the Advanced generation tab."""

    generation_section = context.generation_section
    results_section = context.results_section
    dit_handler = context.dit_handler
    llm_handler = context.llm_handler

    def batch_extract_wrapper(input_folder: str, output_folder: str, *args: Any):
        """Stream folder Extract status updates while processing source audio files."""

        yield from run_batch_extract_processing(
            dit_handler,
            llm_handler,
            input_folder,
            output_folder,
            args,
        )

    generation_section["batch_extract_input_browse_btn"].click(
        fn=select_folder_path,
        inputs=[generation_section["batch_extract_input_folder"]],
        outputs=[generation_section["batch_extract_input_folder"]],
    )
    generation_section["batch_extract_output_browse_btn"].click(
        fn=select_folder_path,
        inputs=[generation_section["batch_extract_output_folder"]],
        outputs=[generation_section["batch_extract_output_folder"]],
    )

    generation_section["batch_extract_btn"].click(
        fn=batch_extract_wrapper,
        inputs=[
            generation_section["batch_extract_input_folder"],
            generation_section["batch_extract_output_folder"],
            *build_generation_run_inputs(generation_section, results_section),
        ],
        outputs=[generation_section["batch_extract_status"]],
    )
    cancel_event = generation_section["batch_extract_cancel_btn"].click(
        fn=None,
        inputs=None,
        outputs=[generation_section["batch_extract_cancel_confirmed_state"]],
        js=BATCH_CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
    )
    cancel_event.then(
        fn=request_full_generation_cancel_from_ui,
        inputs=[generation_section["batch_extract_cancel_confirmed_state"]],
        outputs=[generation_section["batch_extract_status"]],
        queue=False,
        show_progress="hidden",
    )
