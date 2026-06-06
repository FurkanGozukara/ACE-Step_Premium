"""Full-video processing helpers for the Audio Processing tab."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .auto_editor_video import run_auto_editor_video
from .dsp_mastering import measure_lufs
from .media_io import mux_video_with_audio, read_media_audio, save_processed_audio
from .pipeline import ProcessedAudio, process_audio_array
from .process_logging import emit_process_message
from .settings import AudioProcessingSettings


def process_video_with_auto_editor(
    source_video: str | Path,
    input_audio: np.ndarray,
    sample_rate: int,
    output_dir: str | Path,
    output_stem: str,
    settings: AudioProcessingSettings,
    *,
    progress_callback: Any = None,
) -> tuple[ProcessedAudio, str, str]:
    """Process video audio, cut video with Auto-Editor, and export audio/video."""

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    untrimmed_settings = replace(settings, trim_empty_output=False)
    processed = process_audio_array(
        input_audio,
        sample_rate,
        untrimmed_settings,
        progress_callback=progress_callback,
    )
    with tempfile.TemporaryDirectory(prefix="acestep_auto_editor_video_") as temp_dir:
        temp_root = Path(temp_dir)
        emit_process_message(progress_callback, "Saving processed audio for video render.")
        temp_audio_path = save_processed_audio(
            processed.after,
            processed.sample_rate,
            temp_root / "processed_audio.wav",
            "wav",
        )
        temp_video_path = mux_video_with_audio(
            source_video,
            temp_audio_path,
            temp_root / "processed_source.mp4",
            process_callback=progress_callback,
        )
        emit_process_message(progress_callback, "Running Auto-Editor video render.")
        video_path = run_auto_editor_video(
            temp_video_path,
            target_dir / f"{output_stem}.mp4",
            settings.trim_settings(),
            settings.video_reencode,
            process_callback=progress_callback,
        )
    final_audio, final_sample_rate = read_media_audio(
        video_path,
        process_callback=progress_callback,
    )
    emit_process_message(progress_callback, "Saving extracted audio from rendered video.")
    audio_path = save_processed_audio(
        final_audio,
        final_sample_rate,
        target_dir / f"{output_stem}.{_extension(settings.output_format)}",
        settings.output_format,
    )
    return (
        _processed_video_audio(processed, final_audio, final_sample_rate, settings),
        audio_path,
        video_path,
    )


def _processed_video_audio(
    source_processed: ProcessedAudio,
    final_audio: np.ndarray,
    final_sample_rate: int,
    settings: AudioProcessingSettings,
) -> ProcessedAudio:
    """Build processed-audio metrics for an Auto-Editor-rendered video."""

    after = np.asarray(final_audio, dtype=np.float32)
    duration_seconds = len(after) / float(final_sample_rate)
    return ProcessedAudio(
        before=source_processed.before,
        after=after,
        sample_rate=int(final_sample_rate),
        lufs_before=source_processed.lufs_before,
        lufs_after=measure_lufs(after, final_sample_rate),
        duration_seconds=duration_seconds,
        diffpitcher_metadata=source_processed.diffpitcher_metadata,
        trim_metadata=_video_trim_metadata(source_processed, duration_seconds, settings),
    )


def _video_trim_metadata(
    source_processed: ProcessedAudio,
    output_duration_seconds: float,
    settings: AudioProcessingSettings,
) -> dict[str, object]:
    """Return JSON-safe metadata for Auto-Editor video trimming."""

    source_duration = float(source_processed.duration_seconds)
    removed_seconds = max(0.0, source_duration - float(output_duration_seconds))
    applied = removed_seconds > 0.05
    return {
        "enabled": bool(settings.trim_empty_output),
        "applied": applied,
        "reason": "auto_editor_video_trimmed" if applied else "no_trim_needed",
        "mode": "auto_editor_video",
        "source_duration_seconds": source_duration,
        "trimmed_duration_seconds": output_duration_seconds,
        "removed_duration_seconds": removed_seconds,
        "auto_quality": settings.video_reencode.auto_set_quality,
    }


def _extension(output_format: Any) -> str:
    """Return extension for a processed audio output format."""

    normalized = str(output_format or "wav").lower()
    return normalized if normalized in {"wav", "flac", "mp3"} else "wav"
