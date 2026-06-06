"""Event wiring for the Audio Processing tab."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.batch import run_batch_audio_processing
from acestep.audio_processing.presets import PRESET_VALUES, STAGE_KEYS
from acestep.audio_processing.settings import UI_SETTING_KEYS, settings_from_ui_values
from acestep.ui.gradio.events.local_path_dialogs import select_folder_path
from acestep.ui.gradio.events.wiring.audio_processing_cancel_actions import (
    AUDIO_PROCESSING_CANCEL_CONFIRM_JS,
    request_audio_processing_cancel_from_ui,
)
from acestep.ui.gradio.events.wiring.audio_processing_process_status import (
    open_audio_processing_outputs_folder,
)
from acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers import (
    _effective_single_file_input,
    preview_single_file as _preview_single_file,
    preview_upload as _preview_upload,
    process_single_file as _process_single_file,
)
from acestep.ui.gradio.events.wiring.audio_processing_single_file_subprocess import (
    process_single_file_subprocess,
)


def audio_processing_generation_inputs(component_map: dict[str, Any]) -> list[Any]:
    """Return ordered audio-processing controls for generation callbacks."""

    return [component_map[key] for key in UI_SETTING_KEYS]


def register_audio_processing_handlers(audio_page: dict[str, Any]) -> None:
    """Register manual single-file and batch audio-processing handlers."""

    settings_inputs = audio_processing_generation_inputs(audio_page)
    audio_page["ap_builtin_preset"].change(
        fn=_apply_builtin_preset,
        inputs=[audio_page["ap_builtin_preset"]],
        outputs=[audio_page[f"ap_{key}"] for key in STAGE_KEYS],
    )
    audio_page["ap_toggle_audio_enhancement_btn"].click(
        fn=_toggle_audio_enhancement_stages,
        inputs=[audio_page[f"ap_{key}_enabled"] for key in STAGE_KEYS],
        outputs=[audio_page[f"ap_{key}_enabled"] for key in STAGE_KEYS],
        queue=False,
    )
    audio_page["ap_single_file"].change(
        fn=_preview_upload,
        inputs=[audio_page["ap_single_file"]],
        outputs=[
            audio_page["ap_upload_audio_preview"],
            audio_page["ap_upload_video_preview"],
            audio_page["ap_single_status"],
        ],
        queue=False,
    )
    audio_page["ap_preview_btn"].click(
        fn=_preview_single_file,
        inputs=[
            audio_page["ap_single_file"],
            audio_page["ap_upload_audio_preview"],
            *settings_inputs,
        ],
        outputs=[
            audio_page["ap_preview_before_audio"],
            audio_page["ap_preview_after_audio"],
            audio_page["ap_spectrogram"],
            audio_page["ap_single_files"],
            audio_page["ap_single_status"],
        ],
        api_name="audio_processing_preview",
    )
    process_event = audio_page["ap_process_btn"].click(
        fn=_process_single_file_event,
        inputs=[
            audio_page["ap_single_file"],
            audio_page["ap_upload_audio_preview"],
            audio_page["ap_run_subprocess"],
            *settings_inputs,
        ],
        outputs=[
            audio_page["ap_output_audio"],
            audio_page["ap_output_video"],
            audio_page["ap_spectrogram"],
            audio_page["ap_single_files"],
            audio_page["ap_single_status"],
        ],
        api_name="audio_processing_process",
    )
    audio_page["ap_cancel_processing_btn"].click(
        fn=None,
        inputs=None,
        outputs=[audio_page["ap_cancel_confirmed_state"]],
        js=AUDIO_PROCESSING_CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
        cancels=[process_event],
    ).then(
        fn=request_audio_processing_cancel_from_ui,
        inputs=[
            audio_page["ap_cancel_confirmed_state"],
            audio_page["ap_run_subprocess"],
        ],
        outputs=[audio_page["ap_single_status"]],
        queue=False,
        show_progress="hidden",
    )
    audio_page["ap_open_outputs_folder_btn"].click(
        fn=open_audio_processing_outputs_folder,
        outputs=[audio_page["ap_single_status"]],
        queue=False,
    )
    audio_page["ap_batch_input_browse_btn"].click(
        fn=select_folder_path,
        inputs=[audio_page["ap_batch_input_folder"]],
        outputs=[audio_page["ap_batch_input_folder"]],
    )
    audio_page["ap_batch_output_browse_btn"].click(
        fn=select_folder_path,
        inputs=[audio_page["ap_batch_output_folder"]],
        outputs=[audio_page["ap_batch_output_folder"]],
    )
    audio_page["ap_batch_process_btn"].click(
        fn=_process_batch_folder,
        inputs=[
            audio_page["ap_batch_input_folder"],
            audio_page["ap_batch_output_folder"],
            audio_page["ap_batch_recursive"],
            *settings_inputs,
        ],
        outputs=[audio_page["ap_batch_status"], audio_page["ap_batch_files"]],
        api_name="audio_processing_batch",
    )


def _apply_builtin_preset(preset_name: str | None) -> tuple[Any, ...]:
    """Return slider updates for a built-in audio-processing preset."""

    values = PRESET_VALUES.get(str(preset_name or ""), PRESET_VALUES["Generic AI"])
    return tuple(gr.update(value=values[key]) for key in STAGE_KEYS)


def _toggle_audio_enhancement_stages(*enabled_values: Any) -> tuple[Any, ...]:
    """Return updates that toggle all stage checkboxes."""

    target = not all(bool(value) for value in enabled_values)
    return tuple(gr.update(value=target) for _ in STAGE_KEYS)


def _process_single_file_event(
    input_value: Any,
    audio_preview_value: Any,
    run_subprocess: Any,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Route Process File through subprocess or in-process execution."""

    if bool(run_subprocess):
        return process_single_file_subprocess(
            input_value,
            audio_preview_value,
            *settings_values,
            progress=progress,
        )
    return _process_single_file(
        input_value,
        audio_preview_value,
        *settings_values,
        progress=progress,
    )


def _process_batch_folder(
    input_folder: str,
    output_folder: str,
    recursive: bool,
    *settings_values: Any,
):
    """Stream batch-folder processing status and generated files."""

    settings = settings_from_ui_values(settings_values)
    for status, files in run_batch_audio_processing(
        input_folder,
        output_folder,
        bool(recursive),
        settings,
    ):
        yield status, gr.update(value=files, visible=bool(files))
