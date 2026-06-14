"""Small UI action helpers for SAM Audio Segment wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
from loguru import logger

from acestep.audio_processing.media_io import is_video_file
from acestep.sam_audio_segment.cancel import request_sam_audio_cancel
from acestep.sam_audio_segment.settings import SamAudioSettings
from acestep.sam_audio_segment.vram_presets import get_sam_vram_preset
from acestep.ui.gradio.events.training.runtime_cleanup import prepare_parent_runtime_for_training
from acestep.ui.gradio.media_upload_values import latest_upload_path

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


def preview_upload(input_value: Any) -> tuple[Any, Any, str]:
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

    batch_artifacts = artifacts.get("_batch_segment_artifacts")
    if isinstance(batch_artifacts, list) and batch_artifacts:
        return _batch_segment_status(batch_artifacts, cleanup_status)

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
            f"- Trim: `{_trim_label(metadata)}`",
            f"- Metadata: `{artifacts.get('metadata_path')}`",
        ]
    )
    return "\n".join(lines)


def _batch_segment_status(batch_artifacts: list[Any], cleanup_status: str) -> str:
    """Return status markdown for a multi-prompt Batch Segment run."""

    lines = []
    if cleanup_status:
        lines.append(cleanup_status)
    lines.extend(
        [
            "### SAM Audio Batch Segment Output",
            f"- Segments: `{len(batch_artifacts)}`",
        ]
    )
    for index, artifact in enumerate(batch_artifacts, start=1):
        if not isinstance(artifact, dict):
            continue
        metadata = _read_metadata(artifact.get("metadata_path"))
        prompt = str(artifact.get("_batch_segment_prompt") or "").strip()
        if not prompt:
            prompt = _prompt_label(metadata)
        lines.append(
            f"- {index}. `{prompt}` -> `{artifact.get('target_audio_path')}`"
        )
        residual = artifact.get("residual_audio_path")
        if residual:
            lines.append(f"  Remaining: `{residual}`")
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


def _prompt_label(metadata: dict[str, Any]) -> str:
    """Return the prompt description saved in SAM-Audio metadata."""

    prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    if not isinstance(prompt, dict):
        return "segment"
    description = str(prompt.get("description") or "").strip()
    return description or "segment"


def _trim_label(metadata: dict[str, Any]) -> str:
    """Return a compact trim summary from saved SAM-Audio metadata."""

    trim = metadata.get("trim") if isinstance(metadata, dict) else None
    if not isinstance(trim, dict) or not trim.get("enabled"):
        return "disabled"
    original = _format_seconds(trim.get("original_duration_seconds"))
    trimmed = _format_seconds(trim.get("trimmed_duration_seconds"))
    reason = str(trim.get("reason") or "unchanged")
    if trim.get("applied"):
        return f"applied ({original}s -> {trimmed}s, {reason})"
    return f"not applied ({reason})"


def _format_seconds(value: Any) -> str:
    """Return a short seconds string for status labels."""

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def request_sam_audio_cancel_from_ui(
    confirmed: bool,
    subprocess_mode_enabled: bool = False,
) -> str | Any:
    """Request SAM-Audio subprocess cancellation after browser confirmation."""

    if not confirmed:
        return gr.skip()
    if not subprocess_mode_enabled:
        return SAM_IN_PROCESS_STATUS
    had_active_work = request_sam_audio_cancel()
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
        values["long_audio_mode"],
        values["chunk_seconds"],
        values["chunk_overlap_seconds"],
    )
