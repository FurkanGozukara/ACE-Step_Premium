"""Unit tests for training progress-stat formatting."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.training.training_progress_stats import (
    build_training_progress_text,
)


class TrainingProgressStatsTests(unittest.TestCase):
    """Verify progress text exposes useful command-line stats."""

    def test_formats_epoch_eta_done_left_and_speed(self) -> None:
        """Epoch status should collapse to one concise metrics line."""

        text = build_training_progress_text(
            "Epoch 2/10, Step 20, Loss: 0.1234",
            step=20,
            total_epochs=10,
            elapsed_seconds=40,
            total_steps=100,
        )

        self.assertEqual(
            "Elapsed: 40s | Epochs: 2/10 (20.0% done, 8 left) | ETA: ~2m 40s "
            "| Speed: 0.50 it/s - Loss: 0.1234 - Step 20/100",
            text,
        )
        self.assertIn("20.0% done", text)
        self.assertIn("8 left", text)
        self.assertIn("ETA: ~2m 40s", text)
        self.assertIn("Speed: 0.50 it/s", text)
        self.assertNotIn("Epoch 2/10, Step 20", text)

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
        self.assertNotIn("Loss:", text)

    def test_setup_status_does_not_show_zero_loss(self) -> None:
        """Setup messages should not display the trainer's placeholder zero loss."""

        text = build_training_progress_text(
            "Starting training (device: cuda, precision: bf16-mixed)...",
            step=0,
            total_epochs=100,
            elapsed_seconds=1,
            loss=0.0,
        )

        self.assertEqual(
            "Starting training (device: cuda, precision: bf16-mixed)... | Elapsed: 1s",
            text,
        )

    def test_uses_explicit_loss_when_status_has_no_loss(self) -> None:
        """Non-metric messages should keep their text and append current loss."""

        text = build_training_progress_text(
            "Checkpoint saved",
            step=30,
            total_epochs=10,
            elapsed_seconds=60,
            loss=0.5,
            total_steps=100,
        )

        self.assertEqual(
            "Checkpoint saved | Elapsed: 1m 0s | Speed: 0.50 it/s - Loss: 0.5000 "
            "- Step 30/100",
            text,
        )


if __name__ == "__main__":
    unittest.main()
