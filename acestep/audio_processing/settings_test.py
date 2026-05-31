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
        self.assertFalse(settings.stage_enabled("noise"))
        self.assertEqual(-16.0, settings.stage_value("lufs"))
        self.assertEqual(set(STAGE_KEYS), set(settings.values))


if __name__ == "__main__":
    unittest.main()
