"""Tests for premium preset default values."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.premium_preset_defaults import ADDITIONAL_DEFAULT_PRESET_VALUES


class PremiumPresetDefaultsTests(unittest.TestCase):
    """Verify premium preset defaults used when no user preset is loaded."""

    def test_lora_save_best_defaults_to_enabled(self) -> None:
        """LoRA presets should start with Save best enabled."""

        self.assertTrue(ADDITIONAL_DEFAULT_PRESET_VALUES["lora_save_best"])


if __name__ == "__main__":
    unittest.main()
