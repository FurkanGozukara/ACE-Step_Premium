"""Fixed-step midpoint solver for SAM-Audio multi-diffusion."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from .multidiffusion_windows import (
    LatentWindow,
    slice_forward_args,
    soft_window_mask,
)

ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], None]


def midpoint_schedule(ode_opt: dict[str, Any]) -> tuple[float, int]:
    """Return fixed midpoint step size and count from torchdiffeq-style options."""

    options = ode_opt.get("options") if isinstance(ode_opt, dict) else None
    step_size = options.get("step_size") if isinstance(options, dict) else None
    if not isinstance(ode_opt, dict) or ode_opt.get("method") != "midpoint":
        raise ValueError("SAM-Audio multi-diffusion requires midpoint ODE settings.")
    try:
        step = float(step_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("SAM-Audio multi-diffusion requires a fixed step_size.") from exc
    raw_steps = 1.0 / step
    num_steps = round(raw_steps)
    if step <= 0 or num_steps <= 0 or not math.isclose(raw_steps, num_steps):
        raise ValueError("SAM-Audio multi-diffusion step_size must evenly divide 1.0.")
    return step, num_steps


def solve_midpoint_multidiffusion(
    model: Any,
    noise: torch.Tensor,
    forward_args: dict[str, Any],
    windows: list[LatentWindow],
    overlap_frames: int,
    *,
    step_size: float,
    num_steps: int,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> torch.Tensor:
    """Solve the flow ODE while fusing local next states after every step."""

    state = noise
    current_time = 0.0
    for step_idx in range(num_steps):
        _check_cancelled(cancel_callback)
        state = _windowed_midpoint_step(
            model,
            state,
            current_time,
            forward_args,
            windows,
            overlap_frames,
            step_size,
            cancel_callback,
        )
        current_time += step_size
        if progress_callback is not None:
            progress_callback(step_idx + 1, num_steps)
    return state


def _windowed_midpoint_step(
    model: Any,
    state: torch.Tensor,
    time_value: float,
    forward_args: dict[str, Any],
    windows: list[LatentWindow],
    overlap_frames: int,
    step_size: float,
    cancel_callback: CancelCallback | None,
) -> torch.Tensor:
    """Advance every window one midpoint step, then merge local next states."""

    batch_size, total_frames, channels = state.shape
    merged = torch.zeros(batch_size, total_frames, channels, device=state.device)
    weights = torch.zeros(batch_size, total_frames, 1, device=state.device)
    time_start = torch.full(
        (batch_size,),
        time_value,
        device=state.device,
        dtype=torch.float32,
    )
    time_mid = torch.full(
        (batch_size,),
        time_value + (0.5 * step_size),
        device=state.device,
        dtype=torch.float32,
    )
    for window in windows:
        _check_cancelled(cancel_callback)
        window_args = slice_forward_args(forward_args, window.start, window.end)
        local_state = state[:, window.start : window.end, :]
        k1 = model.forward(
            noisy_audio=local_state,
            time=time_start,
            **window_args,
        )
        midpoint = local_state + (0.5 * step_size) * k1
        k2 = model.forward(noisy_audio=midpoint, time=time_mid, **window_args)
        local_next = local_state + step_size * k2
        weight = soft_window_mask(
            window.length,
            overlap_frames,
            start=window.start,
            end=window.end,
            total_frames=total_frames,
            device=state.device,
        ).view(1, -1, 1)
        merged[:, window.start : window.end, :] += local_next.float() * weight
        weights[:, window.start : window.end, :] += weight
    return (merged / weights.clamp_min(1e-6)).to(dtype=state.dtype)


def _check_cancelled(callback: CancelCallback | None) -> None:
    """Invoke an optional cancellation callback."""

    if callback is not None:
        callback()
