"""Auto-editor based trimming for generated and processed audio."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .auto_editor_runner import create_analysis_wav, read_v3_audio_spans, run_auto_editor
from .auto_editor_trim_settings import (
    DEFAULT_AUTO_EDITOR_TRIM_SETTINGS,
    AutoEditorTrimSettings,
)
from .process_logging import ProcessCallback


@dataclass(frozen=True)
class SilenceTrimResult:
    """Trimmed audio tensor and JSON-safe trim metadata."""

    audio: torch.Tensor
    metadata: dict[str, object]


def trim_silent_edges(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    enabled: bool,
    trim_settings: AutoEditorTrimSettings | None = None,
    threshold_db: object | None = None,
    process_callback: ProcessCallback | None = None,
) -> SilenceTrimResult:
    """Cut inactive audio sections with auto-editor and return metadata."""

    sample_count = audio_sample_count(audio)
    settings = trim_settings or DEFAULT_AUTO_EDITOR_TRIM_SETTINGS
    metadata = _base_metadata(enabled, sample_rate, sample_count, settings)
    if not enabled:
        metadata["reason"] = "disabled"
        return SilenceTrimResult(audio=audio, metadata=metadata)
    if sample_count <= 0:
        metadata["reason"] = "empty_audio"
        return SilenceTrimResult(audio=audio, metadata=metadata)

    with tempfile.TemporaryDirectory(prefix="acestep_auto_editor_trim_") as temp_dir:
        spans = _detect_spans_with_auto_editor(
            audio,
            sample_rate=int(sample_rate or 48000),
            settings=settings,
            temp_dir=Path(temp_dir),
            process_callback=process_callback,
        )

    if not spans:
        metadata["reason"] = "no_active_segments"
        return SilenceTrimResult(audio=audio, metadata=metadata)

    trimmed = _concat_spans(audio, spans)
    trimmed_count = audio_sample_count(trimmed)
    if trimmed_count == sample_count and _covers_full_audio(spans, sample_count):
        metadata["reason"] = "already_full_span"
        return SilenceTrimResult(audio=audio, metadata=metadata)

    metadata.update(
        {
            "applied": True,
            "reason": "auto_editor_trimmed",
            "trimmed_samples": trimmed_count,
            "trimmed_duration_seconds": _duration_seconds(trimmed_count, sample_rate),
            "segments": _metadata_spans(spans, sample_rate),
            "segments_count": len(spans),
        }
    )
    return SilenceTrimResult(audio=trimmed, metadata=metadata)


def audio_sample_count(audio: torch.Tensor) -> int:
    """Return the number of time samples in an audio tensor."""

    if audio.ndim == 0:
        return int(audio.numel())
    if audio.ndim == 1:
        return int(audio.shape[0])
    return int(audio.shape[-1])


def _detect_spans_with_auto_editor(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    settings: AutoEditorTrimSettings,
    temp_dir: Path,
    process_callback: ProcessCallback | None = None,
) -> list[tuple[int, int]]:
    """Return source sample spans kept by auto-editor."""

    source_wav = temp_dir / "source.wav"
    analysis_wav = temp_dir / "analysis.wav"
    timeline_path = temp_dir / "timeline.v3"
    sf.write(str(source_wav), _tensor_to_channel_last(audio), sample_rate)
    if settings.normalize_analysis_audio:
        create_analysis_wav(source_wav, analysis_wav, process_callback=process_callback)
    else:
        analysis_wav = source_wav
    try:
        run_auto_editor(
            analysis_wav,
            timeline_path,
            settings,
            process_callback=process_callback,
        )
    except RuntimeError as exc:
        if "Timeline is empty" in str(exc):
            return []
        raise
    return read_v3_audio_spans(timeline_path, sample_rate, audio_sample_count(audio))


def _concat_spans(audio: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Concatenate source sample spans in output order."""

    return torch.cat([audio[..., start:end] for start, end in spans], dim=-1)


def _covers_full_audio(spans: list[tuple[int, int]], sample_count: int) -> bool:
    """Return whether spans cover the full source without cuts."""

    return len(spans) == 1 and spans[0][0] <= 0 and spans[0][1] >= sample_count


def _metadata_spans(spans: list[tuple[int, int]], sample_rate: int) -> list[dict[str, object]]:
    """Return JSON-safe segment metadata."""

    return [
        {
            "start_sample": start,
            "end_sample": end,
            "start_seconds": _duration_seconds(start, sample_rate),
            "end_seconds": _duration_seconds(end, sample_rate),
        }
        for start, end in spans
    ]


def _base_metadata(
    enabled: bool,
    sample_rate: int,
    sample_count: int,
    settings: AutoEditorTrimSettings,
) -> dict[str, object]:
    """Return common JSON-safe trim metadata."""

    duration = _duration_seconds(sample_count, sample_rate)
    return {
        "enabled": bool(enabled),
        "applied": False,
        "mode": "auto_editor",
        "reason": "unchanged",
        "settings": settings.to_payload(),
        "sample_rate": int(sample_rate),
        "original_samples": sample_count,
        "trimmed_samples": sample_count,
        "original_duration_seconds": duration,
        "trimmed_duration_seconds": duration,
    }


def _tensor_to_channel_last(audio: torch.Tensor) -> np.ndarray:
    """Return audio tensor data as channel-last float32 NumPy samples."""

    channel_first = _as_channel_first(audio.detach().float().cpu())
    return channel_first.numpy().T.astype(np.float32, copy=False)


def _as_channel_first(audio: torch.Tensor) -> torch.Tensor:
    """Return a two-dimensional channel-first tensor view."""

    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim == 2:
        return audio
    return audio.reshape(-1, audio.shape[-1])


def _duration_seconds(sample_count: int, sample_rate: int) -> float:
    """Return rounded duration seconds for metadata."""

    return round(float(sample_count) / float(max(1, int(sample_rate))), 6)
