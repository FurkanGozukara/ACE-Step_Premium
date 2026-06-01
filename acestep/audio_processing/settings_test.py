"""Tests for audio-processing settings serialization."""

from __future__ import annotations

import unittest

from acestep.audio_processing.presets import STAGE_KEYS
from acestep.audio_processing.settings import UI_SETTING_KEYS, settings_from_ui_values


class AudioProcessingSettingsTests(unittest.TestCase):
    """Verify UI values map safely into processing settings."""

    def test_settings_from_ui_values_preserves_generated_postprocess_flags(self) -> None:
        """UI setting order should preserve auto post-processing controls."""

        values = []
        for key in UI_SETTING_KEYS:
            if key == "ap_auto_postprocess":
                values.append(True)
            elif key == "ap_preserve_original":
                values.append(False)
            elif key == "ap_output_format":
                values.append("flac")
            elif key == "ap_trim_empty_output":
                values.append(True)
            elif key == "ap_trim_threshold_db":
                values.append(-45.0)
            elif key == "ap_trim_margin_seconds":
                values.append(0.3)
            elif key == "ap_trim_mincut":
                values.append(20)
            elif key == "ap_trim_minclip":
                values.append(4)
            elif key == "ap_builtin_preset":
                values.append("Suno")
            elif key.endswith("_enabled"):
                values.append(key != "ap_noise_enabled")
            else:
                values.append(-16.0 if key == "ap_lufs" else 0.5)

        settings = settings_from_ui_values(values)

        self.assertTrue(settings.enabled)
        self.assertFalse(settings.preserve_original)
        self.assertEqual("flac", settings.output_format)
        self.assertTrue(settings.trim_empty_output)
        self.assertEqual(-45.0, settings.trim_threshold_db)
        self.assertEqual(0.3, settings.trim_margin_seconds)
        self.assertEqual(20, settings.trim_mincut)
        self.assertEqual(4, settings.trim_minclip)
        self.assertFalse(settings.stage_enabled("noise"))
        self.assertEqual(-16.0, settings.stage_value("lufs"))
        self.assertEqual(set(STAGE_KEYS), set(settings.values))

    def test_trim_threshold_is_clamped_to_supported_range(self) -> None:
        """Saved Audio Processing trim thresholds use the shared trim range."""

        values = []
        for key in UI_SETTING_KEYS:
            if key == "ap_trim_threshold_db":
                values.append(120.0)
            else:
                values.append(None)

        settings = settings_from_ui_values(values)

        self.assertEqual(0.0, settings.trim_threshold_db)

        for index, key in enumerate(UI_SETTING_KEYS):
            if key == "ap_trim_threshold_db":
                values[index] = -120.0

        settings = settings_from_ui_values(values)

        self.assertEqual(-100.0, settings.trim_threshold_db)


if __name__ == "__main__":
    unittest.main()
