"""Event wiring for Batch Folder Processing."""

from __future__ import annotations

from typing import Any

from acestep.ui.gradio.events.local_path_dialogs import select_folder_path
from acestep.ui.gradio.events.generation.cancel_actions import (
    BATCH_CANCEL_CONFIRM_JS,
    request_generation_cancel_from_ui,
)
from acestep.ui.gradio.events.batch_folder_runner import run_batch_folder_processing
from acestep.ui.gradio.events.wiring.generation_run_wiring import build_generation_run_inputs


def register_batch_folder_handlers(
    *,
    dit_handler: Any,
    llm_handler: Any,
    batch_section: dict[str, Any],
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
) -> None:
    """Register the batch-folder processing button handler."""

    def batch_wrapper(input_folder, output_folder, auto_improve_lyrics, auto_improve_style, *args):
        """Stream status updates while processing a folder of lyrics files."""

        yield from run_batch_folder_processing(
            dit_handler,
            llm_handler,
            input_folder,
            output_folder,
            auto_improve_lyrics,
            auto_improve_style,
            args,
        )

    batch_section["batch_input_folder_browse_btn"].click(
        fn=select_folder_path,
        inputs=[batch_section["batch_input_folder"]],
        outputs=[batch_section["batch_input_folder"]],
    )
    batch_section["batch_output_folder_browse_btn"].click(
        fn=select_folder_path,
        inputs=[batch_section["batch_output_folder"]],
        outputs=[batch_section["batch_output_folder"]],
    )

    batch_section["batch_process_btn"].click(
        fn=batch_wrapper,
        inputs=[
            batch_section["batch_input_folder"],
            batch_section["batch_output_folder"],
            batch_section["batch_auto_improve_lyrics"],
            batch_section["batch_auto_improve_style"],
            *build_generation_run_inputs(generation_section, results_section),
        ],
        outputs=[batch_section["batch_status"]],
    )
    batch_cancel_event = batch_section["batch_cancel_btn"].click(
        fn=None,
        inputs=None,
        outputs=[batch_section["batch_cancel_confirmed_state"]],
        js=BATCH_CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    batch_cancel_event.then(
        fn=request_generation_cancel_from_ui,
        inputs=[
            batch_section["batch_cancel_confirmed_state"],
            generation_section["subprocess_mode_checkbox"],
        ],
        outputs=[batch_section["batch_status"]],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
