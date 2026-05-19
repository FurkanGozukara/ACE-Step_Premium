"""Tests for auto-label progress timing text."""

from __future__ import annotations

import unittest

from acestep.training.dataset_builder_modules.label_progress import LabelProgressTracker


class _Clock:
    """Controllable monotonic clock for deterministic progress tests."""

    def __init__(self, value: float = 0.0) -> None:
        """Store the initial clock value."""

        self.value = value

    def __call__(self) -> float:
        """Return the current clock value."""

        return self.value


class LabelProgressTrackerTests(unittest.TestCase):
    """Verify auto-label progress includes useful speed and ETA details."""

    def test_start_message_reports_eta_calculating_before_first_completion(self) -> None:
        """Initial progress should keep the existing prefix and show elapsed time."""

        clock = _Clock()
        tracker = LabelProgressTracker(total=2, time_fn=clock)

        tracker.begin_item()
        message = tracker.start_message(1, 0, 2, "a.flac")

        self.assertTrue(message.startswith("Labeling 1/2; labeled 0/2; left 2: a.flac"))
        self.assertIn("elapsed 00:00", message)
        self.assertIn("ETA calculating", message)

    def test_complete_message_reports_last_speed_and_eta(self) -> None:
        """Completed progress should show last duration, average speed, and ETA."""

        clock = _Clock()
        tracker = LabelProgressTracker(total=2, time_fn=clock)

        tracker.begin_item()
        clock.value = 60.0
        tracker.complete_item()
        message = tracker.complete_message(1, 1, 1, "a.flac")

        self.assertTrue(
            message.startswith("Labeling 1/2 complete; labeled 1/2; left 1: a.flac")
        )
        self.assertIn("last 01:00", message)
        self.assertIn("avg 01:00/file", message)
        self.assertIn("speed 1.00/min", message)
        self.assertIn("ETA 01:00", message)


if __name__ == "__main__":
    unittest.main()
