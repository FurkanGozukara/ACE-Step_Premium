"""Media output helpers for SAM-Audio segmentation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from acestep.audio_processing.json_io import write_json
from acestep.audio_processing.media_io import (
    is_video_file,
    mux_video_with_audio,
    save_processed_audio,
)
from acestep.audio_processing.silence_trim import audio_sample_count, trim_silent_edges


@dataclass(frozen=True)
class SamAudioArtifacts:
    """Saved SAM-Audio artifact paths and metadata."""

    source_path: str
    target_audio_path: str
    residual_audio_path: str | None
    target_video_path: str | None
    metadata_path: str
    sample_rate: int
    duration_seconds: float

    def file_list(self) -> list[str]:
        """Return artifact paths for Gradio file outputs."""

        paths = [self.target_audio_path]
        if self.residual_audio_path:
            paths.append(self.residual_audio_path)
        if self.target_video_path:
            paths.append(self.target_video_path)
        paths.append(self.metadata_path)
        return paths


def save_sam_audio_outputs(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    output_stem: str,
    target: torch.Tensor,
    residual: torch.Tensor | None,
    sample_rate: int,
    output_format: str,
    include_residual: bool,
    include_video: bool,
    metadata: dict[str, Any],
    trim_empty_output: bool = False,
    trim_threshold_db: float = -40.0,
) -> SamAudioArtifacts:
    """Save target, residual, optional video mux, and metadata."""

    source = Path(source_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    ext = _extension(output_format)
    trim_result = trim_silent_edges(
        target,
        sample_rate=sample_rate,
        enabled=trim_empty_output,
        threshold_db=trim_threshold_db,
    )
    target_to_save = trim_result.audio
    target_audio = save_processed_audio(
        _tensor_to_audio(target_to_save),
        sample_rate,
        target_dir / f"{output_stem}.{ext}",
        ext,
    )
    residual_audio = None
    if include_residual and residual is not None:
        residual_audio = save_processed_audio(
            _tensor_to_audio(residual),
            sample_rate,
            target_dir / f"{output_stem}_residual.{ext}",
            ext,
        )
    target_video = None
    if include_video and is_video_file(source):
        target_video = mux_video_with_audio(
            source,
            target_audio,
            target_dir / f"{output_stem}.mp4",
        )
    duration = float(audio_sample_count(target_to_save)) / float(sample_rate)
    metadata_path = write_json(
        target_dir / f"{output_stem}.sam_audio.json",
        {
            "_meta": {
                "format": "ace_step_sam_audio_segment",
                "version": 1,
                "source_path": str(source).replace("\\", "/"),
            },
            **metadata,
            "trim": trim_result.metadata,
            "outputs": {
                "target_audio_path": target_audio,
                "residual_audio_path": residual_audio,
                "target_video_path": target_video,
            },
            "metrics": {
                "sample_rate": sample_rate,
                "duration_seconds": duration,
            },
        },
    )
    return SamAudioArtifacts(
        source_path=str(source).replace("\\", "/"),
        target_audio_path=target_audio,
        residual_audio_path=residual_audio,
        target_video_path=target_video,
        metadata_path=metadata_path,
        sample_rate=sample_rate,
        duration_seconds=duration,
    )


def _tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """Return a channel-last float32 NumPy audio array."""

    data = tensor.detach().float().cpu().numpy()
    if data.ndim == 1:
        return data[:, None].astype(np.float32, copy=False)
    return data.T.astype(np.float32, copy=False)


def _extension(value: str) -> str:
    """Return a supported audio extension."""

    normalized = str(value or "wav").strip().lower()
    return normalized if normalized in {"wav", "flac", "mp3"} else "wav"
