"""Small UI action helpers for SAM Audio Segment wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
from loguru import logger

from acestep.core.generation.cancellation import request_generation_cancel
from acestep.audio_processing.media_io import is_video_file
from acestep.sam_audio_segment.settings import SamAudioSettings
from acestep.sam_audio_segment.vram_presets import get_sam_vram_preset
from acestep.ui.gradio.events.training.runtime_cleanup import prepare_parent_runtime_for_training

SAM_CANCEL_CONFIRM_JS = (
    "() => [confirm('Are you sure you want to cancel the current SAM-Audio run?')]"
)
SAM_BATCH_CANCEL_CONFIRM_JS = (
    "() => [confirm('Are you sure you want to cancel the current SAM-Audio batch?')]"
)
SAM_CANCEL_REQUESTED_STATUS = (
    "SAM-Audio subprocess cancellation requested. The isolated worker is being stopped."
)
SAM_NO_ACTIVE_STATUS = "No active SAM-Audio subprocess is currently running."
SAM_IN_PROCESS_STATUS = (
    "Subprocess mode is off. The current in-process SAM run cannot be interrupted safely."
)


def preview_upload(input_path: str | None) -> tuple[Any, Any, str]:
    """Return accurate audio/video preview updates for an uploaded file."""

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


def release_generation_if_requested(
    dit_handler: Any,
    llm_handler: Any,
    settings: SamAudioSettings,
) -> str:
    """Release foreground generation resources before SAM-Audio when requested."""

    if not settings.unload_generation:
        return ""
    return prepare_parent_runtime_for_training(dit_handler, llm_handler, release_dit=True)


def single_status(artifacts: dict[str, Any], cleanup_status: str) -> str:
    """Return single-file status markdown."""

    lines = []
    metadata = _read_metadata(artifacts.get("metadata_path"))
    if cleanup_status:
        lines.append(cleanup_status)
    lines.extend(
        [
            "### SAM Audio Output",
            f"- Model: `{_model_label(metadata)}`",
            f"- Extracted audio: `{artifacts.get('target_audio_path')}`",
            f"- Remaining audio: `{artifacts.get('residual_audio_path') or 'None'}`",
            f"- Extracted video: `{artifacts.get('target_video_path') or 'None'}`",
            f"- Metadata: `{artifacts.get('metadata_path')}`",
        ]
    )
    return "\n".join(lines)


def _read_metadata(path: Any) -> dict[str, Any]:
    """Return saved SAM-Audio metadata when available."""

    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _model_label(metadata: dict[str, Any]) -> str:
    """Return a compact model label from saved metadata."""

    model = metadata.get("model") if isinstance(metadata, dict) else None
    if not isinstance(model, dict):
        return "Unknown"
    path = str(model.get("path") or "")
    name = Path(path).name if path else "Unknown"
    dtype = str(model.get("dtype") or "").replace("torch.", "")
    return f"{name} ({dtype})" if dtype else name


def request_sam_audio_cancel_from_ui(
    confirmed: bool,
    subprocess_mode_enabled: bool = False,
) -> str | Any:
    """Request SAM-Audio subprocess cancellation after browser confirmation."""

    if not confirmed:
        return gr.skip()
    if not subprocess_mode_enabled:
        return SAM_IN_PROCESS_STATUS
    had_active_work = request_generation_cancel(subprocess_only=True)
    if not had_active_work:
        logger.info("[sam_audio_cancel] Cancel requested, but no SAM subprocess is active.")
        return SAM_NO_ACTIVE_STATUS
    logger.info("[sam_audio_cancel] Cancellation requested from UI.")
    return SAM_CANCEL_REQUESTED_STATUS


def apply_vram_preset(preset: str | None) -> tuple[Any, ...]:
    """Return UI updates for a SAM-Audio VRAM preset."""

    values = get_sam_vram_preset(preset)
    return (
        values["quantization"],
        values["attention_backend"],
        values["reranking_candidates"],
        values["ranker_mode"],
        values["predict_spans"],
        values["subprocess"],
        values["ode_steps"],
        values["device_mode"],
        values["low_vram_lite"],
        values["chunked"],
        values["chunk_seconds"],
        values["chunk_overlap_seconds"],
    )
