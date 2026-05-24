"""Progress-stat formatting for LoRA training UI and subprocess logs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_EPOCH_RE = re.compile(r"Epoch\s+(\d+)(?:/(\d+))?")
_LOSS_RE = re.compile(r"Loss:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_RESUME_STEP_RE = re.compile(r"Resumed\s+from\s+epoch\s+\d+,\s+step\s+(\d+)", re.IGNORECASE)
_UNSET = object()
_SPEED_EMA_ALPHA = 0.35


@dataclass(frozen=True)
class TrainingProgressTiming:
    """Timing data derived from optimizer-step progress."""

    elapsed_seconds: float
    elapsed_label: str
    speed: float | None
    eta_seconds: float | None


class TrainingProgressTimer:
    """Track training-only elapsed time, speed, and ETA for UI progress text."""

    def __init__(self, wall_start_time: float) -> None:
        """Create a timer anchored to the UI action start time."""

        self._wall_start_time = wall_start_time
        self._last_setup_time: float | None = None
        self._training_start_time: float | None = None
        self._previous_step: int | None = None
        self._previous_step_time: float | None = None
        self._smoothed_speed: float | None = None
        self._base_step = 0

    def update(
        self,
        status: Any,
        *,
        step: int,
        total_steps: int | None,
        now: float,
    ) -> TrainingProgressTiming:
        """Return timing stats for the latest trainer update."""

        if step <= 0:
            self._last_setup_time = now
            self._update_base_step_from_status(status)
            return TrainingProgressTiming(
                elapsed_seconds=now - self._wall_start_time,
                elapsed_label="Setup",
                speed=None,
                eta_seconds=None,
            )

        if self._training_start_time is None:
            self._training_start_time = self._last_setup_time if self._last_setup_time else now

        training_elapsed = max(0.0, now - self._training_start_time)
        speed = self._update_smoothed_speed(step, now)
        eta = None
        if speed and total_steps and total_steps > step:
            eta = (total_steps - step) / speed

        return TrainingProgressTiming(
            elapsed_seconds=training_elapsed,
            elapsed_label="Elapsed",
            speed=speed,
            eta_seconds=eta,
        )

    def _update_base_step_from_status(self, status: Any) -> None:
        """Capture resumed global-step baseline from setup status messages."""

        match = _RESUME_STEP_RE.search(str(status or ""))
        if match:
            self._base_step = max(0, int(match.group(1)))
            self._previous_step = None
            self._previous_step_time = None
            self._smoothed_speed = None

    def _update_smoothed_speed(self, step: int, now: float) -> float | None:
        """Return recent optimizer-step speed smoothed across progress intervals."""

        if self._previous_step is None or self._previous_step_time is None:
            self._previous_step = max(step, self._base_step)
            self._previous_step_time = now
            return self._smoothed_speed

        delta_steps = int(step) - self._previous_step
        delta_seconds = now - self._previous_step_time
        self._previous_step = int(step)
        self._previous_step_time = now
        if delta_steps <= 0 or delta_seconds <= 0:
            return self._smoothed_speed

        interval_speed = delta_steps / delta_seconds
        if self._smoothed_speed is None:
            self._smoothed_speed = interval_speed
        else:
            self._smoothed_speed = (
                (_SPEED_EMA_ALPHA * interval_speed)
                + ((1.0 - _SPEED_EMA_ALPHA) * self._smoothed_speed)
            )
        return self._smoothed_speed


def build_training_progress_text(
    status: Any,
    *,
    step: int,
    total_epochs: int,
    elapsed_seconds: float,
    loss: float | None = None,
    total_steps: int | None = None,
    elapsed_label: str = "Elapsed",
    speed: float | None | object = _UNSET,
    eta_seconds: float | None | object = _UNSET,
) -> str:
    """Return concise status text with elapsed time, ETA, loss, and step count."""

    status_text = str(status)
    stats = [f"{elapsed_label}: {_format_duration(elapsed_seconds)}"]
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
        eta = (
            _epoch_eta(elapsed_seconds, current_epoch, resolved_total)
            if eta_seconds is _UNSET
            else eta_seconds
        )
        if eta is not None:
            stats.append(f"ETA: ~{_format_duration(eta)}")

    resolved_speed = _step_speed(step, elapsed_seconds) if speed is _UNSET else speed
    if resolved_speed is not None:
        stats.append(f"Speed: {resolved_speed:.2f} it/s")

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
