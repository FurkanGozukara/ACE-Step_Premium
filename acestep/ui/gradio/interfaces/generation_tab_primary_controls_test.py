"""Tests for primary generation-tab controls."""

import unittest

from acestep.ui.gradio.interfaces.generation_tab_primary_controls import (
    DEFAULT_ADVANCED_GENERATION_MODE,
    _default_generation_mode_value,
)


class GenerationTabPrimaryControlsTests(unittest.TestCase):
    """Verify defaults used when the Advanced tab first renders."""

    def test_advanced_mode_defaults_to_remix_when_available(self) -> None:
        """The displayed Remix label should still resolve to the Remix value."""

        choices = [
            "Custom",
            ("Remix (SFT Model Recommended)", "Remix"),
            ("Repaint (SFT Model Recommended)", "Repaint"),
        ]

        self.assertEqual(
            _default_generation_mode_value(choices),
            DEFAULT_ADVANCED_GENERATION_MODE,
        )

    def test_advanced_mode_falls_back_to_custom_without_remix(self) -> None:
        """Unexpected mode lists should keep Custom as the safe fallback."""

        self.assertEqual(
            _default_generation_mode_value(["Custom", "Extract"]),
            "Custom",
        )


if __name__ == "__main__":
    unittest.main()
