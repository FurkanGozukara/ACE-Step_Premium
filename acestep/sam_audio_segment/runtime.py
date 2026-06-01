"""Low-level SAM-Audio runtime loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from safetensors.torch import load_file

from acestep.audio_processing.media_io import read_media_audio


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a SAM-Audio checkpoint from safetensors or torch format."""

    if path.suffix.lower() == ".safetensors":
        return load_file(str(path), device=str(device))
    try:
        payload = torch.load(path, map_location=device, weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported SAM-Audio checkpoint payload: {type(payload)!r}")
    return {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def read_audio_tensor(path: Path, sample_rate: int) -> torch.Tensor:
    """Read media audio as channel-first tensor at ``sample_rate``."""

    audio, source_rate = read_media_audio(path)
    tensor = torch.from_numpy(audio.T).float()
    if int(source_rate) != int(sample_rate):
        tensor = torchaudio.functional.resample(tensor, int(source_rate), int(sample_rate))
    return tensor


def resolve_device(value: str) -> torch.device:
    """Resolve an auto/cuda/cpu device string."""

    requested = str(value or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(device: torch.device) -> torch.dtype:
    """Return the preferred SAM-Audio runtime dtype."""

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32
