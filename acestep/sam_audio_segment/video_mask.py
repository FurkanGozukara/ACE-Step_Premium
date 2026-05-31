"""Video mask loading helpers for SAM-Audio visual prompting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_masked_video_tensor(video_path: Path, mask_path: Path) -> torch.Tensor:
    """Load a masked video tensor without relying on torchcodec."""

    import cv2

    frames = _read_video_frames(video_path)
    masks = _read_video_frames(mask_path, grayscale=True)
    if not masks:
        raise ValueError(f"Mask video has no frames: {mask_path}")
    masked_frames = []
    for index, frame in enumerate(frames):
        mask = masks[min(index, len(masks) - 1)]
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        keep = (mask == 0).astype(frame.dtype)
        masked_frames.append(frame * keep[:, :, None])
    array = torch.from_numpy(np.stack(masked_frames))
    return array.permute(0, 3, 1, 2).contiguous()


def _read_video_frames(path: Path, *, grayscale: bool = False) -> list[Any]:
    """Read video frames as RGB or grayscale arrays."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if grayscale:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Video has no frames: {path}")
    return frames
