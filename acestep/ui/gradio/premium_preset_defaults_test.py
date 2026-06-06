"""Tests for premium preset default values."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.premium_preset_defaults import ADDITIONAL_DEFAULT_PRESET_VALUES
from acestep.ui.gradio.premium_features import DEFAULT_PRESET_VALUES


class PremiumPresetDefaultsTests(unittest.TestCase):
    """Verify premium preset defaults used when no user preset is loaded."""

    def test_lora_save_best_defaults_to_enabled(self) -> None:
        """LoRA presets should start with Save best enabled."""

        self.assertTrue(ADDITIONAL_DEFAULT_PRESET_VALUES["lora_save_best"])
        self.assertEqual(10, ADDITIONAL_DEFAULT_PRESET_VALUES["lora_save_best_after"])
        self.assertEqual(
            5,
            ADDITIONAL_DEFAULT_PRESET_VALUES["lora_save_best_smoothing_window"],
        )
        self.assertEqual(
            0.001,
            ADDITIONAL_DEFAULT_PRESET_VALUES["lora_save_best_min_delta"],
        )
        self.assertEqual(
            "constant",
            ADDITIONAL_DEFAULT_PRESET_VALUES["lora_scheduler_type"],
        )

    def test_extract_and_mp3_defaults(self) -> None:
        """Extract defaults should save MP3 output and use 256k MP3 quality."""

        self.assertEqual("mp3", ADDITIONAL_DEFAULT_PRESET_VALUES["extract_output_format"])
        self.assertEqual("mp3", ADDITIONAL_DEFAULT_PRESET_VALUES["sam_output_format"])
        self.assertEqual("256k", DEFAULT_PRESET_VALUES["mp3_bitrate"])

    def test_audio_processing_subprocess_defaults_to_enabled(self) -> None:
        """Audio Processing should process single files in a cancellable worker by default."""

        self.assertTrue(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_run_subprocess"])
        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_export_audio_only"])


if __name__ == "__main__":
    unittest.main()
