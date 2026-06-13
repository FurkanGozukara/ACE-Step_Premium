"""Settings mappers for Load Metadata UI restore values."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_workflow import AUTO_EDITOR_WORKFLOW_EXPORT_KEY
from acestep.audio_processing.diffpitcher_settings import DIFFPITCHER_UI_KEYS
from acestep.audio_processing.presets import DEFAULT_STAGE_VALUES, STAGE_KEYS
from acestep.audio_processing.settings import AudioProcessingSettings, UI_SETTING_KEYS
from acestep.audio_processing.video_reencode_settings import VIDEO_REENCODE_UI_KEYS
from acestep.sam_audio_segment.settings import SAM_AUDIO_PRESET_KEYS, SamAudioSettings


def audio_processing_ui_values(payload: Any) -> dict[str, Any]:
    """Return Audio Processing UI values from a saved settings payload."""

    settings = AudioProcessingSettings.from_payload(payload).to_payload()
    values = {
        "ap_auto_postprocess": settings["enabled"],
        "ap_preserve_original": settings["preserve_original"],
        "ap_output_format": settings["output_format"],
        "ap_export_audio_only": settings["export_audio_only"],
        "ap_trim_empty_output": settings["trim_empty_output"],
        "ap_trim_threshold_db": settings["trim_threshold_db"],
        "ap_trim_margin_seconds": settings["trim_margin_seconds"],
        "ap_trim_mincut": settings["trim_mincut"],
        "ap_trim_minclip": settings["trim_minclip"],
        AUTO_EDITOR_WORKFLOW_EXPORT_KEY: settings["workflow_export"],
        "ap_builtin_preset": settings["preset"],
    }
    values.update(video_reencode_ui_values(settings["video_reencode"]))
    values.update(diffpitcher_ui_values(settings["diffpitcher"]))
    for key in STAGE_KEYS:
        values[f"ap_{key}_enabled"] = settings["stages_enabled"].get(key, True)
        values[f"ap_{key}"] = settings["values"].get(key, DEFAULT_STAGE_VALUES[key])
    return {key: values.get(key, gr.update()) for key in UI_SETTING_KEYS}


def video_reencode_ui_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Return flattened video-reencode UI values."""

    values = {
        "ap_video_auto_quality": payload.get("auto_set_quality"),
        "ap_video_codec": payload.get("video_codec"),
        "ap_video_bitrate": payload.get("video_bitrate"),
        "ap_video_crf": payload.get("video_crf"),
        "ap_video_preset": payload.get("video_preset"),
        "ap_video_audio_codec": payload.get("audio_codec"),
        "ap_video_audio_bitrate": payload.get("audio_bitrate"),
    }
    return {key: values.get(key, gr.update()) for key in VIDEO_REENCODE_UI_KEYS}


def diffpitcher_ui_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Return flattened DiffPitcher UI values."""

    values = {
        "ap_diffpitcher_enabled": payload.get("enabled"),
        "ap_diffpitcher_mode": payload.get("mode"),
        "ap_diffpitcher_reference_audio": payload.get("reference_audio"),
        "ap_diffpitcher_midi": payload.get("midi_path"),
        "ap_diffpitcher_steps": payload.get("steps"),
        "ap_diffpitcher_shift_semitones": payload.get("shift_semitones"),
        "ap_diffpitcher_mask_with_source": payload.get("mask_with_source"),
        "ap_diffpitcher_device": payload.get("device"),
    }
    return {key: values.get(key, gr.update()) for key in DIFFPITCHER_UI_KEYS}


def sam_audio_ui_values(payload: Any) -> dict[str, Any]:
    """Return SAM Audio UI values from a saved settings payload."""

    settings = SamAudioSettings.from_payload(payload).to_payload()
    values: dict[str, Any] = {}
    for key in SAM_AUDIO_PRESET_KEYS:
        payload_key = key.removeprefix("sam_")
        values[key] = settings.get(payload_key, gr.update())
    return values
