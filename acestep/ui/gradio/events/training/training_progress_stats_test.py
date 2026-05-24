"""Unit tests for training progress-stat formatting."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.training.training_progress_stats import (
    TrainingProgressTimer,
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

    def test_timer_excludes_setup_from_training_speed_and_eta(self) -> None:
        """Training timing should start after setup and use optimizer-step progress."""

        timer = TrainingProgressTimer(wall_start_time=0)
        setup_timing = timer.update(
            "Training 167,772,160 parameters",
            step=0,
            total_steps=10000,
            now=9,
        )
        first_step_timing = timer.update(
            "Epoch 1/200, Step 10, Loss: 1.0420",
            step=10,
            total_steps=10000,
            now=44,
        )
        timing = timer.update(
            "Epoch 1/200, Step 20, Loss: 1.0300",
            step=20,
            total_steps=10000,
            now=56,
        )

        self.assertEqual("Setup", setup_timing.elapsed_label)
        self.assertEqual(9, setup_timing.elapsed_seconds)
        self.assertEqual("Elapsed", first_step_timing.elapsed_label)
        self.assertEqual(35, first_step_timing.elapsed_seconds)
        self.assertIsNone(first_step_timing.speed)
        self.assertIsNone(first_step_timing.eta_seconds)
        self.assertEqual(47, timing.elapsed_seconds)
        self.assertAlmostEqual(10 / 12, timing.speed)
        self.assertAlmostEqual((10000 - 20) / (10 / 12), timing.eta_seconds)

    def test_explicit_timing_overrides_epoch_eta_and_speed(self) -> None:
        """Formatter should use step-timer ETA instead of epoch-count ETA."""

        text = build_training_progress_text(
            "Epoch 1/200, Step 10, Loss: 1.0420",
            step=10,
            total_epochs=200,
            elapsed_seconds=35,
            total_steps=10000,
            speed=10 / 35,
            eta_seconds=(10000 - 10) / (10 / 35),
        )

        self.assertIn("Elapsed: 35s", text)
        self.assertIn("Speed: 0.29 it/s", text)
        self.assertIn("ETA: ~9h 42m", text)
        self.assertNotIn("2h 28m", text)

    def test_timer_uses_resume_step_as_speed_baseline(self) -> None:
        """Resumed runs should not divide the resumed global step by fresh elapsed time."""

        timer = TrainingProgressTimer(wall_start_time=0)
        timer.update("✅ Resumed from epoch 60, step 3000", step=0, total_steps=10000, now=5)
        first_timing = timer.update(
            "Epoch 61/200, Step 3010, Loss: 0.5000",
            step=3010,
            total_steps=10000,
            now=40,
        )
        timing = timer.update(
            "Epoch 61/200, Step 3020, Loss: 0.5000",
            step=3020,
            total_steps=10000,
            now=52,
        )

        self.assertIsNone(first_timing.speed)
        self.assertAlmostEqual(10 / 12, timing.speed)


if __name__ == "__main__":
    unittest.main()
