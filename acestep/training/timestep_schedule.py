"""Timestep schedule helpers for adapter training."""

from __future__ import annotations

from math import isfinite


def build_shifted_timestep_schedule(
    num_inference_steps: int,
    shift: float,
) -> list[float]:
    """Return the shifted timestep schedule used for training samples.

    Args:
        num_inference_steps: Number of non-zero timestep points to sample from.
        shift: Schedule shift factor. Values less than or equal to zero are
            treated as ``1.0`` to avoid an invalid diffusion schedule.

    Returns:
        Descending timestep values without a trailing zero.
    """

    steps = _positive_int(num_inference_steps, 1)
    schedule_shift = _positive_float(shift, 1.0)
    raw = [1.0 - index / steps for index in range(steps)]
    if schedule_shift == 1.0:
        return raw
    return [
        schedule_shift * timestep / (1.0 + (schedule_shift - 1.0) * timestep)
        for timestep in raw
    ]


def _positive_int(value: object, default: int) -> int:
    """Coerce *value* to a positive integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _positive_float(value: object, default: float) -> float:
    """Coerce *value* to a positive finite float."""

    try:
        parsed = float(value)
    except (TypeError, OverflowError, ValueError):
        return default
    if parsed <= 0 or not isfinite(parsed):
        return default
    return parsed
