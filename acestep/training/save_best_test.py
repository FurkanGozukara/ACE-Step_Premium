"""Tests for best-checkpoint metric tracking."""

from __future__ import annotations

import unittest

from acestep.training.save_best import BestMetricTracker


class BestMetricTrackerTests(unittest.TestCase):
    """Verify moving-average and min-delta best tracking."""

    def test_uses_moving_average_window(self) -> None:
        """The smoothed metric should average the configured recent window."""

        tracker = BestMetricTracker(smoothing_window=3, min_delta=0.0)

        tracker.observe(0.9)
        tracker.observe(0.6)
        is_best, smoothed = tracker.observe(0.3)

        self.assertTrue(is_best)
        self.assertAlmostEqual(0.6, smoothed)

    def test_min_delta_filters_tiny_improvements(self) -> None:
        """A metric must improve by more than min_delta to replace best."""

        tracker = BestMetricTracker(smoothing_window=1, min_delta=0.001)

        self.assertTrue(tracker.observe(0.5000)[0])
        self.assertFalse(tracker.observe(0.4995)[0])
        self.assertTrue(tracker.observe(0.4988)[0])


if __name__ == "__main__":
    unittest.main()
