"""Tests for premium preset default values."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.premium_preset_defaults import ADDITIONAL_DEFAULT_PRESET_VALUES
from acestep.ui.gradio.premium_features import DEFAULT_PRESET_VALUES
from acestep.ui.gradio.events.generation.remix_presets import (
    REMIX_PRESET_CHOICES,
    REMIX_PRESET_DEFAULT,
    REMIX_PRESET_DIFFERENT_LYRICS,
    REMIX_PRESET_SAME_LYRICS_BIG_CHANGE,
    REMIX_PRESET_SAME_LYRICS_MEDIUM_CHANGE,
    remix_preset_values,
)


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

    def test_remix_strength_defaults_to_default_preset(self) -> None:
        """Generation presets should default Remix Strength to the Default preset."""

        self.assertEqual(
            remix_preset_values(REMIX_PRESET_DEFAULT)[0],
            DEFAULT_PRESET_VALUES["audio_cover_strength"],
        )

    def test_remix_melody_retention_defaults_to_default_preset(self) -> None:
        """Generation presets should default Remix Melody Retention to Default."""

        self.assertEqual(
            remix_preset_values(REMIX_PRESET_DEFAULT)[1],
            DEFAULT_PRESET_VALUES["cover_noise_strength"],
        )

    def test_remix_preset_defaults_to_default(self) -> None:
        """Generation presets should select Default by default."""

        self.assertEqual(
            REMIX_PRESET_DEFAULT,
            DEFAULT_PRESET_VALUES["remix_preset"],
        )

    def test_remix_preset_order_matches_ui_spec(self) -> None:
        """Remix preset choices should preserve the requested UI order."""

        self.assertEqual(
            (
                REMIX_PRESET_DEFAULT,
                REMIX_PRESET_SAME_LYRICS_BIG_CHANGE,
                REMIX_PRESET_SAME_LYRICS_MEDIUM_CHANGE,
                REMIX_PRESET_DIFFERENT_LYRICS,
            ),
            REMIX_PRESET_CHOICES,
        )

    def test_audio_processing_subprocess_defaults_to_enabled(self) -> None:
        """Audio Processing should process single files in a cancellable worker by default."""

        self.assertTrue(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_run_subprocess"])
        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_export_audio_only"])

    def test_generated_song_postprocessing_defaults_to_opt_in(self) -> None:
        """Generated-song post-processing should stay unchecked by default."""

        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_auto_postprocess"])
        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["sam_auto_postprocess"])
        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["sam_batch_segment"])
        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["sam_batch_overwrite_existing"])
        self.assertEqual(["vocals"], ADDITIONAL_DEFAULT_PRESET_VALUES["sam_prompt_preset"])

    def test_diffpitcher_audio_processing_defaults_are_conservative(self) -> None:
        """DiffPitcher presets should default to an opt-in pitch-fix workflow."""

        self.assertFalse(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_enabled"])
        self.assertEqual("template", ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_mode"])
        self.assertIsNone(
            ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_reference_audio"]
        )
        self.assertIsNone(ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_midi"])
        self.assertEqual(50, ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_steps"])
        self.assertEqual(
            0.0,
            ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_shift_semitones"],
        )
        self.assertTrue(
            ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_mask_with_source"]
        )
        self.assertEqual("auto", ADDITIONAL_DEFAULT_PRESET_VALUES["ap_diffpitcher_device"])


if __name__ == "__main__":
    unittest.main()
