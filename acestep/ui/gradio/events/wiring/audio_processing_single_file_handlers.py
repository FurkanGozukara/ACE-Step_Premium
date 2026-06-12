"""Single-file Audio Processing Gradio handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_workflow import (
    export_auto_editor_workflow,
    workflow_export_enabled,
    workflow_export_label,
)
from acestep.audio_processing.file_processor import metrics_markdown, process_media_file
from acestep.audio_processing.media_io import save_processed_audio
from acestep.audio_processing.plots import make_spectrogram_figure
from acestep.audio_processing.runs import create_audio_processing_run_dir, safe_media_stem
from acestep.audio_processing.settings import settings_from_ui_values
from acestep.ui.gradio.events.wiring.audio_processing_process_status import (
    make_process_log_callback,
    with_process_log,
)
from acestep.ui.gradio.media_upload_values import latest_upload_path


PREVIEW_SECONDS = 60.0


def preview_single_file(
    input_value: Any,
    audio_preview_value: Any,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Process a preview slice for one uploaded media file."""

    input_path = _effective_single_file_input(input_value, audio_preview_value)
    if not input_path:
        return None, None, None, gr.update(visible=False), "Upload an audio or video file first."
    settings = settings_from_ui_values(settings_values)
    if workflow_export_enabled(settings.workflow_export):
        return (
            None,
            None,
            None,
            gr.update(value=None, visible=False),
            (
                "Workflow export selected. Preview skipped; "
                "Process File exports only the workflow file."
            ),
        )
    process_log: list[str] = []
    progress_callback = make_process_log_callback(process_log, progress)
    run_dir = create_audio_processing_run_dir()
    try:
        result = process_media_file(
            input_path,
            run_dir,
            settings,
            max_seconds=PREVIEW_SECONDS,
            output_stem=f"{safe_media_stem(input_path)}_preview",
            include_video=False,
            progress_callback=progress_callback,
        )
        before_path = save_processed_audio(
            result.processed_audio.before,
            result.processed_audio.sample_rate,
            Path(run_dir) / f"{safe_media_stem(input_path)}_preview_before.wav",
            "wav",
        )
        figure = make_spectrogram_figure(
            result.processed_audio.before,
            result.processed_audio.after,
            result.processed_audio.sample_rate,
        )
        files = [before_path, *result.file_list()]
        return (
            before_path,
            result.audio_path,
            figure,
            gr.update(value=files, visible=True),
            with_process_log(metrics_markdown(result), process_log),
        )
    except Exception as exc:
        return None, None, None, gr.update(visible=False), f"Preview failed: {exc}"


def process_single_file(
    input_value: Any,
    audio_preview_value: Any,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Process one complete uploaded media file."""

    input_path = _effective_single_file_input(input_value, audio_preview_value)
    if not input_path:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            "Upload an audio or video file first."
        )
    settings = settings_from_ui_values(settings_values)
    process_log: list[str] = []
    progress_callback = make_process_log_callback(process_log, progress)
    run_dir = create_audio_processing_run_dir()
    try:
        if workflow_export_enabled(settings.workflow_export):
            workflow_path = export_auto_editor_workflow(
                input_path,
                run_dir,
                safe_media_stem(input_path),
                settings.workflow_export,
                settings.trim_settings(),
                process_callback=progress_callback,
            )
            return (
                None,
                gr.update(value=None, visible=False),
                None,
                gr.update(value=[workflow_path], visible=True),
                with_process_log(
                    _workflow_export_markdown(
                        input_path,
                        workflow_path,
                        settings.workflow_export,
                    ),
                    process_log,
                ),
            )
        result = process_media_file(
            input_path,
            run_dir,
            settings,
            progress_callback=progress_callback,
        )
        figure = make_spectrogram_figure(
            result.processed_audio.before,
            result.processed_audio.after,
            result.processed_audio.sample_rate,
        )
        return (
            result.audio_path,
            gr.update(value=result.video_path, visible=bool(result.video_path)),
            figure,
            gr.update(value=result.file_list(), visible=True),
            with_process_log(metrics_markdown(result), process_log),
        )
    except Exception as exc:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            f"Processing failed: {exc}"
        )


def _workflow_export_markdown(input_path: str, workflow_path: str, mode: str) -> str:
    """Return UI status for an Auto-Editor workflow-only export."""

    return "\n".join(
        [
            "### Auto-Editor Workflow Export",
            f"- Source: `{Path(input_path).name}`",
            f"- Workflow: `{workflow_export_label(mode)}`",
            f"- Exported file: `{workflow_path}`",
            "- Processed audio/video: `None`",
        ]
    )


def _effective_single_file_input(input_value: Any, audio_preview_value: Any) -> str | None:
    """Return edited audio-preview input when present, otherwise the upload value."""

    return latest_upload_path(audio_preview_value) or latest_upload_path(input_value)
