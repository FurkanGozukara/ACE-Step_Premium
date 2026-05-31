"""Chunking and overlap-add helpers for long SAM-Audio inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class AudioChunk:
    """One channel-first audio chunk with source sample offsets."""

    start: int
    end: int
    audio: torch.Tensor


def should_process_chunked(audio: torch.Tensor, sample_rate: int, chunk_seconds: float) -> bool:
    """Return whether the audio is longer than the configured chunk window."""

    chunk_samples = seconds_to_samples(chunk_seconds, sample_rate)
    return chunk_samples > 0 and int(audio.shape[-1]) > chunk_samples


def iter_audio_chunks(
    audio: torch.Tensor,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Iterable[AudioChunk]:
    """Yield overlapping channel-first chunks that cover the whole audio."""

    total = int(audio.shape[-1])
    chunk_samples = min(total, seconds_to_samples(chunk_seconds, sample_rate))
    overlap_samples = min(seconds_to_samples(overlap_seconds, sample_rate), chunk_samples // 2)
    step = max(1, chunk_samples - overlap_samples)
    start = 0
    while start < total:
        end = min(total, start + chunk_samples)
        yield AudioChunk(start=start, end=end, audio=audio[..., start:end])
        if end >= total:
            break
        start += step


def overlap_add_chunks(
    chunks: list[tuple[int, int, torch.Tensor]],
    total_samples: int,
    overlap_samples: int,
) -> torch.Tensor:
    """Blend separated chunks into one channel-first tensor."""

    if not chunks:
        raise ValueError("No SAM-Audio chunks were produced.")
    channels = 1 if chunks[0][2].ndim == 1 else int(chunks[0][2].shape[0])
    output = torch.zeros(channels, total_samples, dtype=torch.float32)
    weights = torch.zeros(total_samples, dtype=torch.float32)
    for start, end, chunk in chunks:
        data = _fit_chunk(chunk, end - start)
        if data.ndim == 1:
            data = data.unsqueeze(0)
        weight = _chunk_weight(end - start, overlap_samples, start, end, total_samples)
        output[:, start:end] += data.float().cpu() * weight.unsqueeze(0)
        weights[start:end] += weight
    return output / weights.clamp_min(1e-6).unsqueeze(0)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to at least one audio sample."""

    return max(1, int(round(max(0.001, float(seconds)) * int(sample_rate))))


def _chunk_weight(
    length: int,
    overlap_samples: int,
    start: int,
    end: int,
    total_samples: int,
) -> torch.Tensor:
    """Return linear fade weights for one chunk."""

    weight = torch.ones(length, dtype=torch.float32)
    fade = min(overlap_samples, length // 2)
    if fade <= 0:
        return weight
    if start > 0:
        weight[:fade] *= torch.linspace(0.0, 1.0, fade, dtype=torch.float32)
    if end < total_samples:
        weight[-fade:] *= torch.linspace(1.0, 0.0, fade, dtype=torch.float32)
    return weight


def _fit_chunk(chunk: torch.Tensor, expected_samples: int) -> torch.Tensor:
    """Crop or pad a separated chunk to match its source window."""

    data = chunk.detach().float().cpu()
    if data.ndim > 1 and data.shape[0] > data.shape[-1]:
        data = data.T
    current = int(data.shape[-1])
    if current > expected_samples:
        return data[..., :expected_samples]
    if current == expected_samples:
        return data
    return torch.nn.functional.pad(data, (0, expected_samples - current))
