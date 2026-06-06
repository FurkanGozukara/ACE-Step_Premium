"""DDIM and vocoder inference loop for DiffPitcher."""

from __future__ import annotations

import numpy as np
import torch

from .diffpitcher_constants import DIFFPITCHER_MEL_MAX, DIFFPITCHER_MEL_MIN
from .diffpitcher_runtime import DiffPitcherRuntime
from .process_logging import ProcessCallback, emit_process_message


def run_diffusion(
    runtime: DiffPitcherRuntime,
    source_mel: np.ndarray,
    f0_bins: np.ndarray,
    steps: int,
    *,
    progress_callback: ProcessCallback | None = None,
) -> torch.Tensor:
    """Run the DiffPitcher DDIM denoising loop."""

    from diffusers import DDIMScheduler

    with torch.inference_mode():
        source = torch.from_numpy(source_mel).float().unsqueeze(0).to(runtime.device)
        f0 = torch.from_numpy(fit_frames(f0_bins, source_mel.shape[-1])).float()
        f0 = f0.unsqueeze(0).to(runtime.device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(int(steps), device=runtime.device)
        generator = torch.Generator(device=runtime.device).manual_seed(2024)
        pred = torch.randn(source.shape, generator=generator, device=runtime.device)
        source_x = _minmax_norm_diff(source)
        total_steps = len(scheduler.timesteps)
        emit_process_message(
            progress_callback,
            f"DiffPitcher diffusion: 0/{total_steps} steps complete",
            0.0,
        )
        for step_index, timestep in enumerate(scheduler.timesteps, start=1):
            model_input = scheduler.scale_model_input(pred, timestep)
            model_output = runtime.unet(
                x=model_input,
                mean=source_x,
                f0=f0,
                t=timestep,
                ref=None,
                embed=None,
            )
            pred = scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=pred,
                eta=1,
                generator=generator,
            ).prev_sample
            progress_value = step_index / max(1, total_steps)
            emit_process_message(
                progress_callback,
                (
                    "DiffPitcher diffusion step "
                    f"{step_index}/{total_steps} ({progress_value * 100:.1f}%)"
                ),
                progress_value,
            )
        return _reverse_minmax_norm_diff(pred)


def vocode(runtime: DiffPitcherRuntime, mel: torch.Tensor) -> np.ndarray:
    """Render waveform audio from predicted mel features."""

    with torch.inference_mode():
        audio = runtime.vocoder(mel).detach().cpu().squeeze().clamp(-1, 1)
    return audio.numpy().astype(np.float32)


def fit_frames(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Trim or zero-pad a vector to an exact frame count."""

    vector = np.asarray(values)
    if len(vector) >= target_frames:
        return vector[:target_frames]
    return np.pad(vector, (0, target_frames - len(vector)), mode="constant")


def _minmax_norm_diff(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize log mel features to the DiffPitcher diffusion range."""

    tensor = torch.clip(tensor, DIFFPITCHER_MEL_MIN, DIFFPITCHER_MEL_MAX)
    return 2 * (tensor - DIFFPITCHER_MEL_MIN) / (
        DIFFPITCHER_MEL_MAX - DIFFPITCHER_MEL_MIN
    ) - 1


def _reverse_minmax_norm_diff(tensor: torch.Tensor) -> torch.Tensor:
    """Restore log mel features from the diffusion range."""

    tensor = torch.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1) / 2
    return tensor * (DIFFPITCHER_MEL_MAX - DIFFPITCHER_MEL_MIN) + DIFFPITCHER_MEL_MIN
