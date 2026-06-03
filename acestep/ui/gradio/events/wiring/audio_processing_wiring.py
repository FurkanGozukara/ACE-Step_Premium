"""Event wiring for the Audio Processing tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from acestep.audio_processing.batch import run_batch_audio_processing
from acestep.audio_processing.file_processor import metrics_markdown, process_media_file
from acestep.audio_processing.media_io import save_processed_audio
from acestep.audio_processing.media_io import is_video_file
from acestep.audio_processing.plots import make_spectrogram_figure
from acestep.audio_processing.presets import PRESET_VALUES, STAGE_KEYS
from acestep.audio_processing.runs import create_audio_processing_run_dir, safe_media_stem
from acestep.audio_processing.settings import UI_SETTING_KEYS, settings_from_ui_values
from acestep.ui.gradio.events.local_path_dialogs import select_folder_path
from acestep.ui.gradio.media_upload_values import latest_upload_path


PREVIEW_SECONDS = 60.0


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
        inputs=[audio_page["ap_single_file"], *settings_inputs],
        outputs=[
            audio_page["ap_preview_before_audio"],
            audio_page["ap_preview_after_audio"],
            audio_page["ap_spectrogram"],
            audio_page["ap_single_files"],
            audio_page["ap_single_status"],
        ],
        api_name="audio_processing_preview",
    )
    audio_page["ap_process_btn"].click(
        fn=_process_single_file,
        inputs=[audio_page["ap_single_file"], *settings_inputs],
        outputs=[
            audio_page["ap_output_audio"],
            audio_page["ap_output_video"],
            audio_page["ap_spectrogram"],
            audio_page["ap_single_files"],
            audio_page["ap_single_status"],
        ],
        api_name="audio_processing_process",
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


def _preview_upload(input_value: Any) -> tuple[Any, Any, str]:
    """Return accurate audio/video preview updates for an uploaded file."""

    input_path = latest_upload_path(input_value)
    if not input_path:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Upload an audio or video file first.",
        )
    if is_video_file(input_path):
        return (
            gr.update(value=None, visible=False),
            gr.update(value=input_path, visible=True),
            f"Loaded video: `{Path(input_path).name}`",
        )
    return (
        gr.update(value=input_path, visible=True),
        gr.update(value=None, visible=False),
        f"Loaded audio: `{Path(input_path).name}`",
    )


def _preview_single_file(input_value: Any, *settings_values: Any) -> tuple[Any, ...]:
    """Process a preview slice for one uploaded media file."""

    input_path = latest_upload_path(input_value)
    if not input_path:
        return None, None, None, gr.update(visible=False), "Upload an audio or video file first."
    settings = settings_from_ui_values(settings_values)
    run_dir = create_audio_processing_run_dir()
    try:
        result = process_media_file(
            input_path,
            run_dir,
            settings,
            max_seconds=PREVIEW_SECONDS,
            output_stem=f"{safe_media_stem(input_path)}_preview",
            include_video=False,
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
            metrics_markdown(result),
        )
    except Exception as exc:
        return None, None, None, gr.update(visible=False), f"Preview failed: {exc}"


def _process_single_file(input_value: Any, *settings_values: Any) -> tuple[Any, ...]:
    """Process one complete uploaded media file."""

    input_path = latest_upload_path(input_value)
    if not input_path:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            "Upload an audio or video file first."
        )
    settings = settings_from_ui_values(settings_values)
    run_dir = create_audio_processing_run_dir()
    try:
        result = process_media_file(input_path, run_dir, settings)
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
            metrics_markdown(result),
        )
    except Exception as exc:
        return None, gr.update(visible=False), None, gr.update(visible=False), (
            f"Processing failed: {exc}"
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
