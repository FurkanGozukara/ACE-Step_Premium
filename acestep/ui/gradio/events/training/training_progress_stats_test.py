"""Unit tests for training progress-stat formatting."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.training.training_progress_stats import (
    build_training_progress_text,
)


class TrainingProgressStatsTests(unittest.TestCase):
    """Verify progress text exposes useful command-line stats."""

    def test_formats_epoch_eta_done_left_and_speed(self) -> None:
        """Epoch status should include done percentage, remaining epochs, ETA, and speed."""

        text = build_training_progress_text(
            "Epoch 2/10, Step 20, Loss: 0.1234",
            step=20,
            total_epochs=10,
            elapsed_seconds=40,
        )

        self.assertIn("20.0% done", text)
        self.assertIn("8 left", text)
        self.assertIn("ETA: ~2m 40s", text)
        self.assertIn("Speed: 0.50 steps/s", text)

    def test_handles_non_epoch_status(self) -> None:
        """Non-epoch statuses should still include elapsed time."""

        text = build_training_progress_text(
            "Loading checkpoint",
            step=0,
            total_epochs=10,
            elapsed_seconds=5,
        )

        self.assertIn("Loading checkpoint", text)
        self.assertIn("Elapsed: 5s", text)
        self.assertNotIn("Speed:", text)


if __name__ == "__main__":
    unittest.main()
