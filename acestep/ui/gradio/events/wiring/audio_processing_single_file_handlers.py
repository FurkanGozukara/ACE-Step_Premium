"""Single-file Audio Processing Gradio handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_workflow import (
    export_auto_editor_workflow,
    workflow_export_enabled,
)
from acestep.audio_processing.file_processor import metrics_markdown, process_media_file
from acestep.audio_processing.media_io import save_processed_audio
from acestep.audio_processing.plots import make_spectrogram_figure
from acestep.audio_processing.runs import create_audio_processing_run_dir, safe_media_stem
from acestep.audio_processing.settings import settings_from_ui_values
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
    workflow_media_reference,
    workflow_source_input,
)
from acestep.ui.gradio.events.wiring.audio_processing_workflow_outputs import (
    workflow_export_markdown,
)


PREVIEW_SECONDS = 60.0


def preview_single_file(
    input_value: Any,
    audio_preview_value: Any,
    local_path_value: Any = None,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Process a preview slice for one uploaded or local media file."""

    input_path = effective_single_file_input(
        input_value,
        audio_preview_value,
        local_path_value,
    )
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
    local_path_value: Any = None,
    *settings_values: Any,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, ...]:
    """Process one complete uploaded or local media file."""

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
    process_log: list[str] = []
    progress_callback = make_process_log_callback(process_log, progress)
    run_dir = create_audio_processing_run_dir()
    try:
        if workflow_export_enabled(settings.workflow_export):
            media_reference = workflow_media_reference(input_path, local_path_value)
            workflow_path = export_auto_editor_workflow(
                input_path,
                run_dir,
                safe_media_stem(input_path),
                settings.workflow_export,
                settings.trim_settings(),
                process_callback=progress_callback,
                media_reference=media_reference,
            )
            return (
                hidden_output(),
                hidden_output(),
                hidden_output(),
                visible_output([workflow_path]),
                workflow_export_markdown(
                    input_path,
                    workflow_path,
                    settings.workflow_export,
                    media_reference,
                    local_path_value,
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
            visible_if_present(result.audio_path),
            visible_if_present(result.video_path),
            visible_if_present(figure),
            visible_output(result.file_list()),
            with_process_log(metrics_markdown(result), process_log),
        )
    except Exception as exc:
        return hidden_output(), hidden_output(), hidden_output(), hidden_output(), (
            f"Processing failed: {exc}"
        )


def _process_input_path(
    input_value: Any,
    audio_preview_value: Any,
    local_path_value: Any,
    workflow_export: Any,
) -> str | None:
    """Return the processing source for the selected Audio Processing mode."""

    if workflow_export_enabled(workflow_export):
        return workflow_source_input(input_value, audio_preview_value, local_path_value)
    return effective_single_file_input(input_value, audio_preview_value, local_path_value)
