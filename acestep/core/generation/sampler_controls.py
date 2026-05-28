"""Validation helpers for diffusion sampler controls."""

from __future__ import annotations

from math import isfinite
from typing import Optional, Sequence


def normalize_sampler_shift(value: object, default: float = 1.0) -> float:
    """Return a positive finite timestep shift value.

    Args:
        value: User-supplied timestep shift.
        default: Fallback shift used when ``value`` is invalid.

    Returns:
        Positive finite shift value safe for the diffusion timestep formula.
    """

    try:
        shift = float(value)
    except (TypeError, ValueError):
        return default
    if shift <= 0.0 or not isfinite(shift):
        return default
    return shift


def normalize_sampler_timesteps(
    timesteps: Optional[Sequence[object]],
) -> Optional[list[float]]:
    """Validate and coerce custom timesteps.

    Args:
        timesteps: Optional custom timestep sequence.

    Returns:
        Coerced float timestep list, or ``None`` when no custom schedule was supplied.

    Raises:
        ValueError: If any timestep is non-numeric, non-finite, or outside ``[0, 1]``.
    """

    if timesteps is None:
        return None
    try:
        parsed = [float(timestep) for timestep in timesteps]
    except (TypeError, ValueError) as exc:
        raise ValueError("Custom timesteps must be numeric values in [0, 1].") from exc
    if not parsed:
        return None
    if any(not isfinite(timestep) or timestep < 0.0 or timestep > 1.0 for timestep in parsed):
        raise ValueError("Custom timesteps must be finite values in [0, 1].")
    return parsed
