"""Helpers for best-checkpoint metric tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf


@dataclass
class BestMetricTracker:
    """Track the best moving-average loss with a minimum improvement delta."""

    smoothing_window: int = 5
    min_delta: float = 0.001
    best_metric: float = inf
    recent_metrics: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize tracker settings."""

        self.smoothing_window = max(1, int(self.smoothing_window))
        self.min_delta = max(0.0, float(self.min_delta))

    def observe(self, metric: float) -> tuple[bool, float]:
        """Record a metric and return whether it is a new best.

        Args:
            metric: Raw training or validation loss for the current epoch.

        Returns:
            ``(is_new_best, smoothed_metric)`` where ``smoothed_metric`` is the
            moving average over the configured recent window.
        """

        self.recent_metrics.append(float(metric))
        if len(self.recent_metrics) > self.smoothing_window:
            self.recent_metrics.pop(0)

        smoothed = sum(self.recent_metrics) / len(self.recent_metrics)
        if smoothed < self.best_metric - self.min_delta:
            self.best_metric = smoothed
            return True, smoothed
        return False, smoothed
