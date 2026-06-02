"""Flow-edit parameter normalization helpers."""

from __future__ import annotations

import math
from typing import Any


def normalize_flow_edit_n_avg(value: Any) -> int:
    """Return a valid flow-edit Monte Carlo sample count.

    Args:
        value: User-provided UI/API value for ``n_avg``.

    Returns:
        An integer sample count clamped to at least ``1``.
    """

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(parsed):
        return 1
    return max(1, int(math.ceil(parsed)))
