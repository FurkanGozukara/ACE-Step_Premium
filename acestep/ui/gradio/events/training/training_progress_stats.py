"""Progress-stat formatting for LoRA training UI and subprocess logs."""

from __future__ import annotations

import re
from typing import Any


_EPOCH_RE = re.compile(r"Epoch\s+(\d+)(?:/(\d+))?")
_LOSS_RE = re.compile(r"Loss:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def build_training_progress_text(
    status: Any,
    *,
    step: int,
    total_epochs: int,
    elapsed_seconds: float,
    loss: float | None = None,
    total_steps: int | None = None,
) -> str:
    """Return concise status text with elapsed time, ETA, loss, and step count."""

    status_text = str(status)
    stats = [f"Elapsed: {_format_duration(elapsed_seconds)}"]
    epoch_info = _parse_epoch(status_text, total_epochs)
    status_has_metric = _is_metric_status(status_text, epoch_info)
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
        stats.append(f"Speed: {speed:.2f} it/s")

    suffixes = []
    resolved_loss = _resolve_loss(status_text, loss if step > 0 else None)
    if resolved_loss is not None:
        suffixes.append(f"Loss: {resolved_loss:.4f}")
    if step > 0:
        suffixes.append(_format_step(step, total_steps))

    progress_text = " | ".join(stats)
    if suffixes:
        progress_text = f"{progress_text} - {' - '.join(suffixes)}"
    if status_has_metric:
        return progress_text
    return f"{status_text} | {progress_text}"


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


def _resolve_loss(status_text: str, loss: float | None) -> float | None:
    """Return the current loss from the handler value or trainer status text."""

    if loss is not None and loss == loss:
        return float(loss)
    match = _LOSS_RE.search(status_text)
    if not match:
        return None
    return float(match.group(1))


def _format_step(step: int, total_steps: int | None) -> str:
    """Format current and total optimizer steps."""

    if total_steps is None or total_steps <= 0:
        return f"Step {step}"
    return f"Step {step}/{total_steps}"


def _is_metric_status(status_text: str, epoch_info: tuple[int, int] | None) -> bool:
    """Return whether the original trainer status is redundant metric text."""

    return epoch_info is not None or _LOSS_RE.search(status_text) is not None


def _format_duration(seconds: float) -> str:
    """Format seconds as compact human-readable text."""

    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60}s"
    return f"{total // 3600}h {(total % 3600) // 60}m"
