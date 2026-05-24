"""Training dataset-load and preprocess wiring helpers."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.events.local_path_dialogs import (
    select_folder_path,
    select_optional_json_file_path,
)

from .. import training_handlers as train_h
from ..training.subprocess_cancel import (
    PREPROCESS_CANCEL_CONFIRM_JS,
    request_preprocess_cancel_from_ui,
)
from ..training.subprocess_dataset import run_preprocess_subprocess
from .context import TrainingWiringContext
from .training_dataset_vram_payloads import (
    build_preprocess_dit_init_payload,
    should_run_dataset_action_in_subprocess,
)
from .training_dataset_load_outputs import (
    build_dataset_load_outputs,
    no_dataset_load_outputs,
)


def _browse_and_load_dataset_json(
    current_path: str,
    builder_state: Any,
) -> tuple[Any, ...]:
    """Pick a dataset JSON file and load it into the dataset builder."""

    selected = select_optional_json_file_path(current_path)
    if selected is None:
        return (gr.update(), *no_dataset_load_outputs("No dataset JSON selected."))

    load_outputs = train_h.load_existing_dataset_for_preprocess(selected, builder_state)
    return (gr.update(value=selected), *load_outputs)


def register_training_dataset_load_handler(
    context: TrainingWiringContext,
    *,
    button_key: str,
    path_key: str,
    status_key: str,
    browse_key: str | None = None,
) -> None:
    """Register one dataset JSON load button with shared output/update contracts."""

    training_section = context.training_section
    training_section[button_key].click(
        fn=train_h.load_existing_dataset_for_preprocess,
        inputs=[
            training_section[path_key],
            training_section["dataset_builder_state"],
        ],
        outputs=build_dataset_load_outputs(training_section, status_key),
    )

    if browse_key:
        browse_outputs = [
            training_section[path_key],
            *build_dataset_load_outputs(training_section, status_key),
        ]
        training_section[browse_key].click(
            fn=_browse_and_load_dataset_json,
            inputs=[
                training_section[path_key],
                training_section["dataset_builder_state"],
            ],
            outputs=browse_outputs,
        )


def register_training_preprocess_handler(context: TrainingWiringContext) -> None:
    """Register preprocess button wiring for tensor conversion."""

    training_section = context.training_section
    dit_handler = context.dit_handler
    training_section["preprocess_output_dir_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["preprocess_output_dir"]],
        outputs=[training_section["preprocess_output_dir"]],
    )

    def run_preprocess(
        output_dir,
        mode,
        state,
        model,
        vram_preset,
        subprocess_mode,
        progress=gr.Progress(track_tqdm=True),
    ):
        """Run preprocessing in-process or in an isolated worker."""

        if should_run_dataset_action_in_subprocess(vram_preset, subprocess_mode):
            return run_preprocess_subprocess(
                output_dir=output_dir,
                preprocess_mode=mode,
                builder_state=state,
                model_config=model,
                dit_init_params=build_preprocess_dit_init_payload(
                    dit_handler,
                    model,
                    vram_preset,
                ),
                progress=progress,
            )
        return train_h.preprocess_dataset(
            output_dir,
            mode,
            dit_handler,
            state,
            model_config=model,
        )

    training_section["preprocess_btn"].click(
        fn=run_preprocess,
        inputs=[
            training_section["preprocess_output_dir"],
            training_section["preprocess_mode"],
            training_section["dataset_builder_state"],
            training_section["dataset_model_config"],
            training_section["dataset_vram_preset"],
            training_section["preprocess_subprocess"],
        ],
        outputs=[training_section["preprocess_progress"]],
    )

    training_section["cancel_preprocess_btn"].click(
        fn=request_preprocess_cancel_from_ui,
        inputs=None,
        outputs=[training_section["preprocess_progress"]],
        js=PREPROCESS_CANCEL_CONFIRM_JS,
        queue=False,
        concurrency_limit=None,
        show_progress="hidden",
    )
