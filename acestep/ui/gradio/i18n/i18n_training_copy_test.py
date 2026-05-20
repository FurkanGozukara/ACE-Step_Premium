"""Tests for training UI translation copy."""

import unittest

from acestep.ui.gradio.i18n import I18n


class I18nTrainingCopyTests(unittest.TestCase):
    """Verify training control help text stays accurate."""

    def test_training_shift_help_is_not_turbo_only(self) -> None:
        """Training Shift help should describe current Gradio training behavior."""

        i18n = I18n(default_language="en")
        help_text = i18n.t("training.shift_info")

        self.assertIn("current Gradio value", help_text)
        self.assertIn("training starts", help_text)
        self.assertNotEqual("Timestep shift for Turbo models.", help_text)

    def test_training_timestep_steps_help_mentions_current_value(self) -> None:
        """Training step-count help should say the submitted UI value is used."""

        i18n = I18n(default_language="en")
        help_text = i18n.t("training.num_inference_steps_info")

        self.assertIn("current Gradio value", help_text)


if __name__ == "__main__":
    unittest.main()
