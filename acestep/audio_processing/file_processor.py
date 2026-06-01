"""Single-file audio and video processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_io import write_json
from .media_io import (
    is_video_file,
    mux_video_with_audio,
    read_media_audio,
    save_processed_audio,
)
from .pipeline import ProcessedAudio, process_audio_array
from .runs import safe_media_stem
from .settings import AudioProcessingSettings


@dataclass(frozen=True)
class ProcessedMedia:
    """Processed media artifact paths and metering.

    Args:
        source_path: Original media path.
        audio_path: Processed audio file path.
        video_path: Processed video file path when a video source was muxed.
        metadata_path: JSON metadata sidecar path.
        processed_audio: In-memory processed audio data and metrics.

    Returns:
        Immutable artifact summary.
    """

    source_path: str
    audio_path: str
    video_path: str | None
    metadata_path: str
    processed_audio: ProcessedAudio

    def file_list(self) -> list[str]:
        """Return generated files suitable for Gradio file outputs."""

        paths = [self.audio_path, self.metadata_path]
        if self.video_path:
            paths.insert(1, self.video_path)
        return paths


def process_media_file(
    input_path: str | Path,
    output_dir: str | Path,
    settings: AudioProcessingSettings,
    *,
    max_seconds: float | None = None,
    output_stem: str | None = None,
    include_video: bool = True,
    progress_callback=None,
) -> ProcessedMedia:
    """Process one audio or video file into an output folder.

    Args:
        input_path: Source audio or video path.
        output_dir: Destination folder.
        settings: Audio processing settings.
        max_seconds: Optional preview trim duration.
        output_stem: Optional target filename stem.
        include_video: Whether to create processed video output for video input.
        progress_callback: Optional processing progress callback.

    Returns:
        Processed media artifact metadata.
    """

    source = Path(input_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or f"{safe_media_stem(source)}_processed"
    audio, sample_rate = read_media_audio(source)
    processed = process_audio_array(
        audio,
        sample_rate,
        settings,
        max_seconds=max_seconds,
        progress_callback=progress_callback,
    )
    audio_target = target_dir / f"{stem}.{_extension(settings.output_format)}"
    audio_path = save_processed_audio(
        processed.after,
        processed.sample_rate,
        audio_target,
        settings.output_format,
    )
    video_path = None
    if include_video and max_seconds is None and is_video_file(source):
        video_path = mux_video_with_audio(source, audio_path, target_dir / f"{stem}.mp4")
    metadata_path = _write_metadata(source, settings, processed, audio_path, video_path)
    return ProcessedMedia(
        source_path=str(source).replace("\\", "/"),
        audio_path=audio_path,
        video_path=video_path,
        metadata_path=metadata_path,
        processed_audio=processed,
    )


def metrics_markdown(result: ProcessedMedia) -> str:
    """Return compact before/after processing metrics for the UI."""

    processed = result.processed_audio
    before = _format_lufs(processed.lufs_before)
    after = _format_lufs(processed.lufs_after)
    delta = ""
    if before != "N/A" and after != "N/A":
        delta = f" ({processed.lufs_after - processed.lufs_before:+.1f} dB)"
    return "\n".join(
        [
            "### Processing Metrics",
            f"- Source: `{Path(result.source_path).name}`",
            f"- Duration: `{processed.duration_seconds:.2f}s`",
            f"- LUFS: `{before}` -> `{after}`{delta}",
            f"- Processed audio: `{result.audio_path}`",
            f"- Processed video: `{result.video_path or 'None'}`",
        ]
    )


def _write_metadata(
    source: Path,
    settings: AudioProcessingSettings,
    processed: ProcessedAudio,
    audio_path: str,
    video_path: str | None,
) -> str:
    """Write a JSON sidecar for one processed file."""

    return write_json(
        Path(audio_path).with_suffix(".audio_processing.json"),
        {
            "_meta": {
                "format": "ace_step_audio_processing",
                "version": 1,
                "source_path": str(source).replace("\\", "/"),
            },
            "settings": settings.to_payload(),
            "metrics": {
                "sample_rate": processed.sample_rate,
                "duration_seconds": processed.duration_seconds,
                "lufs_before": processed.lufs_before,
                "lufs_after": processed.lufs_after,
            },
            "trim": processed.trim_metadata,
            "outputs": {
                "audio_path": audio_path,
                "video_path": video_path,
            },
        },
    )


def _extension(output_format: Any) -> str:
    """Return extension for a processed audio output format."""

    normalized = str(output_format or "wav").lower()
    return normalized if normalized in {"wav", "flac", "mp3"} else "wav"


def _format_lufs(value: float) -> str:
    """Return a display-safe LUFS value."""

    return f"{value:.1f}" if value not in (float("inf"), float("-inf")) and value == value else "N/A"
