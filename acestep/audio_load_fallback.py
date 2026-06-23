"""Audio loading fallback helpers for TorchCodec-backed torchaudio builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger


def load_audio_with_torchaudio_fallback(
    audio_path: str | Path,
    *,
    context: str = "audio load",
) -> tuple[torch.Tensor, int]:
    """Load audio as channel-first tensor with FFmpeg fallback.

    Modern torchaudio routes ``torchaudio.load`` through TorchCodec. On Windows,
    that can fail when the matching TorchCodec/FFmpeg DLLs are unavailable. This
    helper preserves the torchaudio fast path but logs and falls back to the
    shared media decoder when TorchCodec fails.
    """

    source = str(audio_path)
    torchaudio_error: BaseException | None = None
    try:
        import torchaudio

        return torchaudio.load(source)
    except Exception as exc:
        torchaudio_error = exc
        logger.warning(
            "[{}] torchaudio.load failed for '{}': {}. Trying FFmpeg/soundfile fallback.",
            context,
            source,
            _compact_error(exc),
        )

    try:
        from acestep.audio_processing.media_io import read_media_audio

        audio_np, sample_rate = read_media_audio(source)
        audio_tensor = torch.from_numpy(audio_np.T.copy()).to(torch.float32)
        logger.warning(
            "[{}] FFmpeg/soundfile fallback succeeded for '{}'.",
            context,
            source,
        )
        return audio_tensor, int(sample_rate)
    except Exception as fallback_error:
        logger.error(
            "[{}] Audio load failed for '{}'. torchaudio error: {} | fallback error: {}",
            context,
            source,
            _compact_error(torchaudio_error),
            _compact_error(fallback_error),
        )
        raise RuntimeError(
            f"Cannot read '{source}': torchaudio.load failed "
            f"({_compact_error(torchaudio_error)}); FFmpeg/soundfile fallback failed "
            f"({_compact_error(fallback_error)})."
        ) from fallback_error


def _compact_error(exc: Any) -> str:
    if exc is None:
        return "unknown error"
    text = str(exc).strip().splitlines()
    detail = text[0] if text else exc.__class__.__name__
    return f"{exc.__class__.__name__}: {detail}"
