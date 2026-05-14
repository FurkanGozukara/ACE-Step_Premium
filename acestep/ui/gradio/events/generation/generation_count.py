"""Helpers for Gradio's sequential song-count control."""

from __future__ import annotations

from typing import Any


def normalize_generation_count(value: Any, default: int = 1) -> int:
    """Return a positive sequential song count for the Gradio UI.

    Args:
        value: Raw UI value from the Songs control.
        default: Fallback value when the input cannot be parsed.

    Returns:
        An integer greater than or equal to ``1``.
    """

    try:
        count = int(float(value))
    except (TypeError, ValueError, OverflowError):
        count = int(default)
    return max(1, count)


def generation_count_info() -> str:
    """Return the user-facing help text for the Songs control."""

    return "Number of songs to generate sequentially."


def seed_for_generation_index(
    seed: Any,
    generation_index: int,
    *,
    random_seed: bool,
) -> list[int] | None:
    """Return the per-run seed list for one sequential generation.

    Args:
        seed: Raw seed UI value.
        generation_index: Zero-based sequential generation index.
        random_seed: Whether the UI random-seed checkbox is enabled.

    Returns:
        ``None`` for random-seed mode, otherwise a one-item seed list. A
        missing or negative seed stays ``[-1]`` so the existing backend random
        fallback remains intact.
    """

    if random_seed:
        return None

    base_seed = _parse_first_seed(seed)
    if base_seed is None or base_seed < 0:
        return [-1]
    return [base_seed + generation_index]


def _parse_first_seed(seed: Any) -> int | None:
    """Parse the first seed value from legacy scalar or comma-separated input."""

    if seed is None:
        return None
    if isinstance(seed, (int, float)):
        return int(seed)

    text = str(seed).strip()
    if not text:
        return None
    first = text.split(",", 1)[0].strip()
    if not first:
        return None
    try:
        return int(float(first))
    except (TypeError, ValueError, OverflowError):
        return None
