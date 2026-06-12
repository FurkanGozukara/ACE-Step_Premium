"""Audio Processing single-file subprocess UI helpers."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_workflow import workflow_export_enabled
from acestep.audio_processing.cancel import AudioProcessingCancelled
from acestep.audio_processing.runs import create_audio_processing_run_dir, safe_media_stem
from acestep.audio_processing.settings import settings_from_ui_values
from acestep.audio_processing.subprocess_runner import run_audio_processing_subprocess
from acestep.ui.gradio.events.wiring.audio_processing_output_updates import (
    hidden_output,
    visible_if_present,
    visible_output,
)
from acestep.ui.gradio.events.wiring.audio_processing_process_status import (
    make_process_log_callback,
    with_process_log,
)
from acestep.ui.gradio.events.wiring.audio_processing_source_paths import (
    effective_single_file_input,
    local_media_path,
    workflow_media_reference,
    workflow_source_input,
)
from acestep.ui.gradio.events.wiring.audio_processing_workflow_outputs import (
    workflow_export_markdown,
)


def process_single_file_subprocess(
    input_value: Any,
    audio_preview_value: Any,
    local_path_value: Any = None,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Process one uploaded or local media file in an isolated subprocess."""

    settings = settings_from_ui_values(settings_values)
    input_path = _process_input_path(
        input_value,
        audio_preview_value,
        local_path_value,
        settings.workflow_export,
    )
    if not input_path:
        return hidden_output(), hidden_output(), hidden_output(), hidden_output(), (
            "Upload an audio or video file first."
        )
    media_reference = workflow_media_reference(input_path, local_path_value)
    media_reference_is_local = bool(local_media_path(local_path_value))
    process_log: list[str] = []
    progress_callback = make_process_log_callback(process_log, progress)
    run_dir = create_audio_processing_run_dir()
    source_stem = safe_media_stem(input_path)
    output_stem = source_stem if workflow_export_enabled(
        settings.workflow_export
    ) else f"{source_stem}_processed"
    try:
        result = run_audio_processing_subprocess(
            {
                "input_path": input_path,
                "output_dir": str(run_dir),
                "output_stem": output_stem,
                "media_reference_path": media_reference,
                "media_reference_is_local": media_reference_is_local,
                "settings": settings.to_payload(),
            },
            progress_callback=progress_callback,
        )
    except AudioProcessingCancelled:
        return hidden_output(), hidden_output(), hidden_output(), hidden_output(), (
            "Audio Processing cancelled."
        )
    except Exception as exc:
        return hidden_output(), hidden_output(), hidden_output(), hidden_output(), (
            f"Processing failed: {exc}"
        )
    if workflow_export_enabled(settings.workflow_export):
        return _workflow_outputs(input_path, result, settings.workflow_export)
    return _media_outputs(result, process_log)


def _workflow_outputs(
    input_path: str,
    result: dict[str, Any],
    workflow_export: str,
) -> tuple[Any, ...]:
    """Return Gradio outputs for workflow-only subprocess results."""

    workflow_path = str(result.get("workflow_path") or "")
    media_reference = str(result.get("media_reference_path") or input_path)
    local_path_value = media_reference if result.get("media_reference_is_local") else None
    return (
        hidden_output(),
        hidden_output(),
        hidden_output(),
        visible_if_present([workflow_path] if workflow_path else []),
        workflow_export_markdown(
            input_path,
            workflow_path,
            workflow_export,
            media_reference,
            local_path_value,
        ),
    )


def _media_outputs(result: dict[str, Any], process_log: list[str]) -> tuple[Any, ...]:
    """Return Gradio outputs for processed media subprocess results."""

    video_path = result.get("video_path")
    files = list(result.get("files", []) or [])
    return (
        visible_if_present(result.get("audio_path")),
        visible_if_present(video_path),
        visible_if_present(result.get("figure")),
        visible_if_present(files),
        with_process_log(str(result.get("status_markdown") or ""), process_log),
    )


def _process_input_path(
    input_value: Any,
    audio_preview_value: Any,
    local_path_value: Any,
    workflow_export: Any,
) -> str | None:
    """Return the processing source for the selected subprocess mode."""

    if workflow_export_enabled(workflow_export):
        return workflow_source_input(input_value, audio_preview_value, local_path_value)
    return effective_single_file_input(input_value, audio_preview_value, local_path_value)
