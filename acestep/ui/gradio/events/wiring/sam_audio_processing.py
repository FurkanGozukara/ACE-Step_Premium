"""Processing callbacks for the SAM Audio Segment tab."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.trim_ui_settings import (
    AUTO_EDITOR_TRIM_UI_KEYS,
    trim_settings_from_ui_values,
)
from acestep.core.generation.cancellation import GenerationCancelled
from acestep.sam_audio_segment.batch import run_batch_sam_audio
from acestep.sam_audio_segment.paths import create_run_dir, safe_media_stem
from acestep.sam_audio_segment.progress import ProgressCallback
from acestep.sam_audio_segment.service_cache import cached_sam_audio_service
from acestep.sam_audio_segment.settings import (
    SAM_AUDIO_PRESET_KEYS,
    SamAudioSettings,
    settings_from_ui_values,
    with_auto_editor_trim_settings,
)
from acestep.sam_audio_segment.subprocess_runner import run_sam_audio_subprocess
from acestep.ui.gradio.media_upload_values import latest_upload_path

from .sam_audio_action_helpers import release_generation_if_requested, single_status
from .sam_audio_processing_streams import (
    stream_batch_subprocess,
    stream_single_subprocess,
)
from .sam_audio_status_log import SamAudioStatusLog


def process_single_file(
    dit_handler: Any,
    llm_handler: Any,
    input_path: Any,
    audio_preview_value: Any,
    mask_video_path: Any,
    *settings_values: Any,
    progress: Any | None = None,
) -> Any:
    """Process one uploaded file with SAM-Audio."""

    input_path = _effective_single_file_input(input_path, audio_preview_value)
    mask_video_path = latest_upload_path(mask_video_path)
    if not input_path:
        yield None, None, gr.update(visible=False), gr.update(visible=False), (
            "Upload an audio or video file first."
        )
        return
    settings = _settings_from_input_values(settings_values)
    status_log = SamAudioStatusLog(progress)
    status_log.callback(0.0, "Preparing SAM-Audio request")
    cleanup_status = release_generation_if_requested(dit_handler, llm_handler, settings)
    run_dir = create_run_dir()
    if settings.subprocess:
        yield from stream_single_subprocess(
            lambda: _process_single_subprocess(
                input_path,
                mask_video_path,
                run_dir,
                settings,
                status_log.callback,
            ),
            cleanup_status,
            status_log,
        )
        return
    try:
        artifacts, files = _process_single_in_process(
            input_path,
            mask_video_path,
            run_dir,
            settings,
            _progress_callback(progress),
        )
        status = single_status(artifacts, cleanup_status)
        yield (
            artifacts.get("target_audio_path"),
            artifacts.get("residual_audio_path"),
            gr.update(
                value=artifacts.get("target_video_path"),
                visible=bool(artifacts.get("target_video_path")),
            ),
            gr.update(value=files, visible=True),
            status,
        )
    except GenerationCancelled:
        yield None, None, gr.update(visible=False), gr.update(visible=False), (
            "SAM-Audio cancelled."
        )
    except Exception as exc:
        yield None, None, gr.update(visible=False), gr.update(visible=False), (
            f"SAM-Audio failed: {exc}"
        )


def process_batch_folder(
    dit_handler: Any,
    llm_handler: Any,
    input_folder: str,
    output_folder: str,
    recursive: bool,
    *settings_values: Any,
    progress: Any | None = None,
):
    """Stream or run SAM-Audio batch-folder processing."""

    status_log = SamAudioStatusLog(progress)
    settings = _settings_from_input_values(settings_values)
    cleanup_status = release_generation_if_requested(dit_handler, llm_handler, settings)
    if settings.subprocess:
        yield from stream_batch_subprocess(
            lambda: run_sam_audio_subprocess(
                {
                    "mode": "batch",
                    "input_folder": input_folder,
                    "output_folder": output_folder,
                    "recursive": bool(recursive),
                    "settings": settings.to_payload(),
                },
                progress_callback=status_log.callback,
            ),
            cleanup_status,
            status_log,
        )
        return

    for status, files in run_batch_sam_audio(
        input_folder,
        output_folder,
        bool(recursive),
        settings,
        progress_callback=_progress_callback(progress),
    ):
        if cleanup_status:
            status = cleanup_status + "\n" + status
        yield status, gr.update(value=files, visible=bool(files))


def _process_single_subprocess(
    input_path: str,
    mask_video_path: str | None,
    run_dir,
    settings,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, Any], list[str]]:
    """Run one SAM-Audio file in a subprocess and return artifacts."""

    result = run_sam_audio_subprocess(
        {
            "mode": "single",
            "input_path": input_path,
            "output_dir": str(run_dir),
            "output_stem": f"{safe_media_stem(input_path)}_sam",
            "mask_video_path": mask_video_path,
            "settings": settings.to_payload(),
        },
        progress_callback=progress_callback,
    )
    return result["artifacts"], result["files"]


def _process_single_in_process(
    input_path: str,
    mask_video_path: str | None,
    run_dir,
    settings,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, Any], list[str]]:
    """Run one SAM-Audio file inside the Gradio process and return artifacts."""

    with cached_sam_audio_service(
        settings,
        progress_callback=progress_callback,
    ) as service:
        artifact_obj = service.process_file(
            input_path,
            run_dir,
            output_stem=f"{safe_media_stem(input_path)}_sam",
            mask_video_path=mask_video_path,
        )
    return artifact_obj.__dict__, artifact_obj.file_list()


def _progress_callback(progress: Any | None) -> ProgressCallback | None:
    """Return a SAM-Audio progress callback backed by Gradio progress."""

    if progress is None:
        return None

    def _report(fraction: float, message: str) -> None:
        progress(float(fraction), desc=str(message))

    return _report


def _settings_from_input_values(values: tuple[Any, ...]) -> SamAudioSettings:
    """Build SAM-Audio settings and apply shared Audio Processing trim values."""

    sam_count = len(SAM_AUDIO_PRESET_KEYS)
    trim_count = len(AUTO_EDITOR_TRIM_UI_KEYS)
    settings = settings_from_ui_values(values[:sam_count])
    trim_values = values[sam_count : sam_count + trim_count]
    if len(trim_values) == trim_count:
        return with_auto_editor_trim_settings(
            settings,
            trim_settings_from_ui_values(trim_values),
        )
    return settings


def _effective_single_file_input(input_value: Any, audio_preview_value: Any) -> str | None:
    """Return edited audio-preview input when present, otherwise the upload value."""

    return latest_upload_path(audio_preview_value) or latest_upload_path(input_value)
