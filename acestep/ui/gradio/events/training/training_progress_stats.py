"""Progress-stat formatting for LoRA training UI and subprocess logs."""

from __future__ import annotations

import re
from typing import Any


_EPOCH_RE = re.compile(r"Epoch\s+(\d+)(?:/(\d+))?")


def build_training_progress_text(
    status: Any,
    *,
    step: int,
    total_epochs: int,
    elapsed_seconds: float,
) -> str:
    """Return status text with elapsed time, ETA, epoch progress, and speed."""

    status_text = str(status)
    stats = [f"Elapsed: {_format_duration(elapsed_seconds)}"]
    epoch_info = _parse_epoch(status_text, total_epochs)
    if epoch_info is not None:
        current_epoch, resolved_total = epoch_info
        left_epochs = max(resolved_total - current_epoch, 0)
        percent = (current_epoch / resolved_total) * 100 if resolved_total else 0.0
        stats.append(
            f"Epochs: {current_epoch}/{resolved_total} "
            f"({percent:.1f}% done, {left_epochs} left)"
        )
        eta = _epoch_eta(elapsed_seconds, current_epoch, resolved_total)
        if eta is not None:
            stats.append(f"ETA: ~{_format_duration(eta)}")

    speed = _step_speed(step, elapsed_seconds)
    if speed is not None:
        stats.append(f"Speed: {speed:.2f} steps/s")

    return f"{status_text}\n" + " | ".join(stats)


def _parse_epoch(status_text: str, default_total: int) -> tuple[int, int] | None:
    """Return the current and total epoch count parsed from trainer status."""

    match = _EPOCH_RE.search(status_text)
    if not match:
        return None
    current = max(int(match.group(1)), 0)
    total = int(match.group(2)) if match.group(2) else int(default_total or 0)
    return current, max(total, current, 1)


def _epoch_eta(elapsed_seconds: float, current_epoch: int, total_epochs: int) -> float | None:
    """Estimate remaining seconds from completed epoch progress."""

    if current_epoch <= 0 or total_epochs <= current_epoch:
        return None
    return (elapsed_seconds / current_epoch) * (total_epochs - current_epoch)


def _step_speed(step: int, elapsed_seconds: float) -> float | None:
    """Return average steps per second when enough data is available."""

    if step <= 0 or elapsed_seconds <= 0:
        return None
    return step / elapsed_seconds


def _format_duration(seconds: float) -> str:
    """Format seconds as compact human-readable text."""

    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60}s"
    return f"{total // 3600}h {(total % 3600) // 60}m"
