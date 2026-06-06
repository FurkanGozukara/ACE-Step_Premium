"""Audio Processing single-file subprocess UI helpers."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_workflow import workflow_export_enabled
from acestep.audio_processing.cancel import AudioProcessingCancelled
from acestep.audio_processing.runs import create_audio_processing_run_dir, safe_media_stem
from acestep.audio_processing.settings import settings_from_ui_values
from acestep.audio_processing.subprocess_runner import run_audio_processing_subprocess
from acestep.ui.gradio.events.wiring.audio_processing_process_status import (
    make_process_log_callback,
    with_process_log,
)
from acestep.ui.gradio.media_upload_values import latest_upload_path

from .audio_processing_single_file_handlers import _workflow_export_markdown


def process_single_file_subprocess(
    input_value: Any,
    audio_preview_value: Any,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Process one uploaded media file in an isolated subprocess."""

    input_path = latest_upload_path(audio_preview_value) or latest_upload_path(input_value)
    if not input_path:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            "Upload an audio or video file first."
        )
    settings = settings_from_ui_values(settings_values)
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
                "settings": settings.to_payload(),
            },
            progress_callback=progress_callback,
        )
    except AudioProcessingCancelled:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            "Audio Processing cancelled."
        )
    except Exception as exc:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            f"Processing failed: {exc}"
        )
    if workflow_export_enabled(settings.workflow_export):
        return _workflow_outputs(input_path, result, settings.workflow_export, process_log)
    return _media_outputs(result, process_log)


def _workflow_outputs(
    input_path: str,
    result: dict[str, Any],
    workflow_export: str,
    process_log: list[str],
) -> tuple[Any, ...]:
    """Return Gradio outputs for workflow-only subprocess results."""

    workflow_path = str(result.get("workflow_path") or "")
    return (
        None,
        gr.update(value=None, visible=False),
        None,
        gr.update(value=[workflow_path], visible=bool(workflow_path)),
        with_process_log(
            _workflow_export_markdown(input_path, workflow_path, workflow_export),
            process_log,
        ),
    )


def _media_outputs(result: dict[str, Any], process_log: list[str]) -> tuple[Any, ...]:
    """Return Gradio outputs for processed media subprocess results."""

    video_path = result.get("video_path")
    files = list(result.get("files", []) or [])
    return (
        result.get("audio_path"),
        gr.update(value=video_path, visible=bool(video_path)),
        result.get("figure"),
        gr.update(value=files, visible=bool(files)),
        with_process_log(str(result.get("status_markdown") or ""), process_log),
    )
