"""Latent-window helpers for SAM-Audio multi-diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LatentWindow:
    """A source window in SAM-Audio latent frame indices."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of latent frames in the window."""

        return self.end - self.start


def latent_window_frames(
    *,
    sample_rate: int,
    hop_length: int,
    window_seconds: float,
    overlap_seconds: float,
) -> tuple[int, int]:
    """Return latent-frame window and overlap lengths."""

    frames_per_second = float(sample_rate) / float(hop_length)
    window_frames = max(1, int(round(float(window_seconds) * frames_per_second)))
    overlap_frames = max(0, int(round(float(overlap_seconds) * frames_per_second)))
    return window_frames, min(overlap_frames, window_frames // 2)


def iter_latent_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
) -> list[LatentWindow]:
    """Return overlapping latent windows covering the full sequence."""

    if total_frames <= 0:
        raise ValueError("SAM-Audio multi-diffusion requires at least one frame.")
    window = max(1, min(total_frames, int(window_frames)))
    overlap = max(0, min(int(overlap_frames), window // 2))
    step = max(1, window - overlap)
    starts = list(range(0, max(1, total_frames - window + 1), step))
    tail_start = max(0, total_frames - window)
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return [
        LatentWindow(start=start, end=min(total_frames, start + window))
        for start in starts
    ]


def soft_window_mask(
    length: int,
    overlap_frames: int,
    *,
    start: int,
    end: int,
    total_frames: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return an edge-aware triangular soft mask for one latent window."""

    weight = torch.ones(length, dtype=torch.float32, device=device)
    fade = min(max(0, int(overlap_frames)), length // 2)
    if fade <= 0:
        return weight
    if start > 0:
        weight[:fade] *= torch.linspace(
            0.0,
            1.0,
            fade,
            dtype=torch.float32,
            device=device,
        )
    if end < total_frames:
        weight[-fade:] *= torch.linspace(
            1.0,
            0.0,
            fade,
            dtype=torch.float32,
            device=device,
        )
    return weight


def slice_forward_args(
    forward_args: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Return SAM-Audio forward args cropped to one latent window."""

    return {
        "audio_features": forward_args["audio_features"][:, start:end, :],
        "text_features": forward_args["text_features"],
        "text_mask": forward_args["text_mask"],
        "masked_video_features": forward_args["masked_video_features"][:, :, start:end],
        "anchor_ids": forward_args["anchor_ids"],
        "anchor_alignment": forward_args["anchor_alignment"][:, start:end],
        "audio_pad_mask": forward_args["audio_pad_mask"][:, start:end],
    }
