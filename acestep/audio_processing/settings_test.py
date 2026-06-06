"""Tests for audio-processing settings serialization."""

from __future__ import annotations

import unittest

from acestep.audio_processing.diffpitcher_settings import DiffPitcherSettings
from acestep.audio_processing.presets import STAGE_KEYS
from acestep.audio_processing.settings import (
    AudioProcessingSettings,
    UI_SETTING_KEYS,
    settings_from_ui_values,
)


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
            elif key == "ap_export_audio_only":
                values.append(True)
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
            elif key == "ap_auto_editor_workflow_export":
                values.append("resolve")
            elif key == "ap_video_auto_quality":
                values.append(False)
            elif key == "ap_video_codec":
                values.append("libx265")
            elif key == "ap_video_bitrate":
                values.append("6000k")
            elif key == "ap_video_crf":
                values.append(20)
            elif key == "ap_video_preset":
                values.append("slow")
            elif key == "ap_video_audio_codec":
                values.append("aac")
            elif key == "ap_video_audio_bitrate":
                values.append("256k")
            elif key == "ap_diffpitcher_enabled":
                values.append(True)
            elif key == "ap_diffpitcher_mode":
                values.append("score")
            elif key == "ap_diffpitcher_reference_audio":
                values.append({"path": "C:/guide/reference.wav"})
            elif key == "ap_diffpitcher_midi":
                values.append({"name": "C:/guide/score.mid"})
            elif key == "ap_diffpitcher_steps":
                values.append(60)
            elif key == "ap_diffpitcher_shift_semitones":
                values.append(-1.5)
            elif key == "ap_diffpitcher_mask_with_source":
                values.append(False)
            elif key == "ap_diffpitcher_device":
                values.append("cpu")
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
        self.assertTrue(settings.export_audio_only)
        self.assertTrue(settings.trim_empty_output)
        self.assertEqual(-45.0, settings.trim_threshold_db)
        self.assertEqual(0.3, settings.trim_margin_seconds)
        self.assertEqual(20, settings.trim_mincut)
        self.assertEqual(4, settings.trim_minclip)
        self.assertEqual("resolve", settings.workflow_export)
        self.assertFalse(settings.video_reencode.auto_set_quality)
        self.assertEqual("libx265", settings.video_reencode.video_codec)
        self.assertEqual("6000k", settings.video_reencode.video_bitrate)
        self.assertEqual(20, settings.video_reencode.video_crf)
        self.assertEqual("slow", settings.video_reencode.video_preset)
        self.assertEqual("aac", settings.video_reencode.audio_codec)
        self.assertEqual("256k", settings.video_reencode.audio_bitrate)
        self.assertTrue(settings.diffpitcher.enabled)
        self.assertEqual("score", settings.diffpitcher.mode)
        self.assertEqual("C:/guide/reference.wav", settings.diffpitcher.reference_audio)
        self.assertEqual("C:/guide/score.mid", settings.diffpitcher.midi_path)
        self.assertEqual(60, settings.diffpitcher.steps)
        self.assertEqual(-1.5, settings.diffpitcher.shift_semitones)
        self.assertFalse(settings.diffpitcher.mask_with_source)
        self.assertEqual("cpu", settings.diffpitcher.device)
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

    def test_audio_processing_payload_round_trips_diffpitcher_settings(self) -> None:
        """Saved Audio Processing payloads should preserve DiffPitcher options."""

        original = AudioProcessingSettings(
            diffpitcher=DiffPitcherSettings(
                enabled=True,
                mode="score",
                reference_audio="C:/guide/ref.wav",
                midi_path="C:/guide/score.mid",
                steps=75,
                shift_semitones=2.0,
                mask_with_source=False,
                device="cpu",
            )
        )

        restored = AudioProcessingSettings.from_payload(original.to_payload())

        self.assertEqual(original.diffpitcher, restored.diffpitcher)


if __name__ == "__main__":
    unittest.main()
