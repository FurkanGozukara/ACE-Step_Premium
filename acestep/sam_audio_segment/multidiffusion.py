"""Text-only SAM-Audio multi-diffusion orchestration for long inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from .chunking import overlap_add_chunks
from .multidiffusion_solver import (
    CancelCallback,
    ProgressCallback,
    midpoint_schedule,
    solve_midpoint_multidiffusion,
)
from .multidiffusion_windows import (
    iter_latent_windows,
    latent_window_frames,
)


@dataclass(frozen=True)
class MultiDiffusionResult:
    """SAM-Audio separation output plus the number of fused latent windows."""

    target: list[torch.Tensor]
    residual: list[torch.Tensor]
    noise: torch.Tensor
    window_count: int


def separate_text_multidiffusion(
    model: Any,
    batch: Any,
    *,
    ode_opt: dict[str, Any],
    window_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> MultiDiffusionResult:
    """Separate one text-conditioned batch by fusing overlapping latent windows.

    Args:
        model: Loaded SAM-Audio model exposing the official private helpers.
        batch: SAM-Audio processor batch for one audio item.
        ode_opt: Fixed midpoint solver options.
        window_seconds: Window size used to crop latent audio features.
        overlap_seconds: Context overlap used for soft mask blending.
        sample_rate: Audio sample rate for seconds-to-frame conversion.
        progress_callback: Optional callback receiving completed and total ODE steps.
        cancel_callback: Optional cancellation checker called during the solve.

    Returns:
        MultiDiffusionResult with target/residual tensors matching ``model.separate``.
    """

    if int(batch.audios.size(0)) != 1:
        raise ValueError("SAM-Audio multi-diffusion currently supports one audio item.")

    step_size, num_steps = midpoint_schedule(ode_opt)
    forward_args = model._get_forward_args(batch, candidates=1)
    audio_features = forward_args["audio_features"]
    _, total_frames, _ = audio_features.shape
    window_frames, overlap_frames = latent_window_frames(
        sample_rate=sample_rate,
        hop_length=int(batch.hop_length),
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
    )
    windows = iter_latent_windows(total_frames, window_frames, overlap_frames)
    logger.info(
        "[sam_audio] Processing {} multi-diffusion latent windows "
        "(window={} frames, overlap={} frames)",
        len(windows),
        window_frames,
        overlap_frames,
    )

    noise = torch.randn_like(audio_features)
    denoised = solve_midpoint_multidiffusion(
        model,
        noise,
        forward_args,
        windows,
        overlap_frames,
        step_size=step_size,
        num_steps=num_steps,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    return _decode_result(model, batch, denoised, noise, windows, overlap_frames)


def _decode_result(
    model: Any,
    batch: Any,
    denoised: torch.Tensor,
    noise: torch.Tensor,
    windows: list[Any],
    overlap_frames: int,
) -> MultiDiffusionResult:
    """Decode latent target/residual features using the official output logic."""

    if len(windows) > 1:
        return _decode_windowed_result(model, batch, denoised, noise, windows, overlap_frames)

    batch_size, total_frames, doubled_channels = denoised.shape
    channels = doubled_channels // 2
    generated_features = denoised.transpose(1, 2)
    wavs = model.audio_codec.decode(
        generated_features.reshape(2 * batch_size, channels, total_frames)
    ).view(batch_size, 2, -1)
    sizes = model.audio_codec.feature_idx_to_wav_idx(batch.sizes)
    target_wavs = model.unbatch(wavs[:, 0].view(batch_size, 1, -1), sizes)
    residual_wavs = model.unbatch(wavs[:, 1].view(batch_size, 1, -1), sizes)
    idxs = torch.zeros(batch_size, dtype=torch.long, device=noise.device)
    return MultiDiffusionResult(
        target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
        residual=[wav[idx] for wav, idx in zip(residual_wavs, idxs, strict=False)],
        noise=noise,
        window_count=len(windows),
    )


def _decode_windowed_result(
    model: Any,
    batch: Any,
    denoised: torch.Tensor,
    noise: torch.Tensor,
    windows: list[Any],
    overlap_frames: int,
) -> MultiDiffusionResult:
    """Decode long latent outputs in overlapping chunks to avoid decoder OOM."""

    _, total_frames, doubled_channels = denoised.shape
    channels = doubled_channels // 2
    generated_features = denoised.transpose(1, 2)
    total_samples = int(model.audio_codec.feature_idx_to_wav_idx(batch.sizes)[0])
    overlap_samples = int(model.audio_codec.feature_idx_to_wav_idx(overlap_frames))
    target_chunks: list[tuple[int, int, torch.Tensor]] = []
    residual_chunks: list[tuple[int, int, torch.Tensor]] = []
    for window in windows:
        start_sample = int(model.audio_codec.feature_idx_to_wav_idx(window.start))
        end_sample = int(model.audio_codec.feature_idx_to_wav_idx(window.end))
        window_features = generated_features[:, :, window.start : window.end]
        wavs = model.audio_codec.decode(
            window_features.reshape(2, channels, window.length)
        ).view(1, 2, -1)
        target_chunks.append((start_sample, end_sample, wavs[0, 0]))
        residual_chunks.append((start_sample, end_sample, wavs[0, 1]))
        del wavs, window_features
        if generated_features.device.type == "cuda":
            torch.cuda.empty_cache()
    target = overlap_add_chunks(target_chunks, total_samples, overlap_samples)
    residual = overlap_add_chunks(residual_chunks, total_samples, overlap_samples)
    return MultiDiffusionResult(
        target=[target[0]],
        residual=[residual[0]],
        noise=noise,
        window_count=len(windows),
    )
