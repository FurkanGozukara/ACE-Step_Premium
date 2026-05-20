"""Tests for model-aware training schedule defaults."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.training.schedule_defaults import (
    training_schedule_defaults_for_model,
    training_schedule_updates_for_model,
)


class TrainingScheduleDefaultsTests(unittest.TestCase):
    """Verify model changes update visible training schedule values."""

    def test_turbo_defaults_to_eight_steps(self) -> None:
        """Turbo training defaults should match the fast distilled schedule."""

        defaults = training_schedule_defaults_for_model("acestep-v15-xl-turbo")

        self.assertEqual(3.0, defaults["shift"])
        self.assertEqual(8, defaults["num_inference_steps"])
        self.assertEqual(200, defaults["num_inference_steps_maximum"])

    def test_sft_defaults_to_fifty_steps(self) -> None:
        """SFT training defaults should expose the longer schedule."""

        defaults = training_schedule_defaults_for_model("acestep-v15-xl-sft")

        self.assertEqual(1.0, defaults["shift"])
        self.assertEqual(50, defaults["num_inference_steps"])
        self.assertEqual(200, defaults["num_inference_steps_maximum"])

    def test_base_defaults_to_training_metadata_values(self) -> None:
        """Base training defaults should use corrected metadata values."""

        defaults = training_schedule_defaults_for_model("C:\\models\\acestep-v15-xl-base")

        self.assertEqual(1.0, defaults["shift"])
        self.assertEqual(50, defaults["num_inference_steps"])
        self.assertEqual(200, defaults["num_inference_steps_maximum"])

    def test_updates_are_gradio_value_updates(self) -> None:
        """The change handler should update both visible controls."""

        shift_update, steps_update = training_schedule_updates_for_model(
            "/models/acestep-v15-xl-base"
        )

        self.assertEqual(1.0, shift_update["value"])
        self.assertEqual(0.1, shift_update["step"])
        self.assertEqual(50, steps_update["value"])
        self.assertEqual(200, steps_update["maximum"])


if __name__ == "__main__":
    unittest.main()
