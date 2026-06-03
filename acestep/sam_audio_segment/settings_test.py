"""Tests for SAM-Audio settings parsing."""

import unittest
from unittest.mock import patch

from acestep.sam_audio_segment.settings import (
    SAM_AUDIO_MAX_CHUNK_SECONDS,
    SAM_AUDIO_PRESET_KEYS,
    SamAudioSettings,
    settings_from_ui_values,
)


class TestSamAudioSettings(unittest.TestCase):
    """Verify UI setting normalization for SAM-Audio."""

    def test_defaults_from_empty_values(self):
        """Empty UI values return conservative defaults."""

        settings = settings_from_ui_values([])

        self.assertFalse(settings.auto_postprocess)
        self.assertTrue(settings.subprocess)
        self.assertEqual(99, settings.seed)
        self.assertFalse(settings.random_seed)
        self.assertEqual("auto", settings.attention_backend)
        self.assertEqual("chunked", settings.long_audio_mode)
        self.assertEqual(20.0, settings.chunk_seconds)
        self.assertEqual(5.0, settings.chunk_overlap_seconds)
        self.assertFalse(settings.trim_empty_output)
        self.assertEqual(-40.0, settings.trim_threshold_db)
        self.assertEqual("mp3", settings.output_format)
        self.assertEqual("vocals", settings.effective_prompt)

    def test_values_follow_schema_order(self):
        """Ordered UI values map to the expected settings fields."""

        values = {key: None for key in SAM_AUDIO_PRESET_KEYS}
        values.update(
            {
                "sam_auto_postprocess": True,
                "sam_preserve_original": True,
                "sam_trim_empty_output": True,
                "sam_trim_threshold_db": -42.0,
                "sam_output_format": "flac",
                "sam_prompt_mode": "span",
                "sam_prompt_preset": "vocals",
                "sam_custom_prompt": "lead vocal",
                "sam_use_span_anchor": True,
                "sam_anchor_json": "",
                "sam_anchor_polarity": "+",
                "sam_anchor_start": 2.0,
                "sam_anchor_end": 3.0,
                "sam_predict_spans": False,
                "sam_reranking_candidates": 2,
                "sam_ranker_mode": "none",
                "sam_ode_steps": 8,
                "sam_seed": 42,
                "sam_random_seed": True,
                "sam_vram_preset": "24gb_balanced",
                "sam_quantization": "none",
                "sam_attention_backend": "cudnn",
                "sam_device_mode": "auto",
                "sam_low_vram_lite": False,
                "sam_chunked": True,
                "sam_long_audio_mode": "multidiffusion",
                "sam_chunk_seconds": 10.0,
                "sam_chunk_overlap_seconds": 1.0,
                "sam_subprocess": True,
                "sam_unload_generation": True,
                "sam_include_residual": True,
                "sam_include_video": True,
                "sam_batch_input_folder": "in",
                "sam_batch_output_folder": "out",
                "sam_batch_recursive": True,
            }
        )

        settings = settings_from_ui_values([values[key] for key in SAM_AUDIO_PRESET_KEYS])

        self.assertTrue(settings.auto_postprocess)
        self.assertEqual("flac", settings.output_format)
        self.assertEqual("lead vocal", settings.effective_prompt)
        self.assertEqual(42, settings.seed)
        self.assertTrue(settings.random_seed)
        self.assertEqual("cudnn", settings.attention_backend)
        self.assertEqual("multidiffusion", settings.long_audio_mode)
        self.assertTrue(settings.trim_empty_output)
        self.assertEqual(-42.0, settings.trim_threshold_db)
        self.assertTrue(settings.batch_recursive)

    def test_trim_threshold_is_clamped_to_safe_range(self):
        """Saved trim thresholds outside the UI range are clamped."""

        quiet = SamAudioSettings.from_payload({"trim_threshold_db": -120.0})
        aggressive = SamAudioSettings.from_payload({"trim_threshold_db": 120.0})

        self.assertEqual(-100.0, quiet.trim_threshold_db)
        self.assertEqual(0.0, aggressive.trim_threshold_db)

    def test_chunk_seconds_accept_long_manual_values(self):
        """Manual chunk inputs above the preset default remain available."""

        settings = SamAudioSettings.from_payload(
            {
                "vram_preset": "32gb_quality",
                "chunk_seconds": 40.0,
                "chunk_overlap_seconds": 2.0,
            }
        )

        self.assertEqual(40.0, settings.chunk_seconds)

    def test_chunk_seconds_still_has_input_sanity_limit(self):
        """Extreme chunk inputs are clamped to the UI sanity range."""

        settings = SamAudioSettings.from_payload(
            {
                "vram_preset": "32gb_quality",
                "chunk_seconds": 600.0,
            }
        )

        self.assertEqual(SAM_AUDIO_MAX_CHUNK_SECONDS, settings.chunk_seconds)

    def test_unavailable_ranker_mode_falls_back_to_disabled(self):
        """Saved ranker values should not crash when optional packages are absent."""

        with patch(
            "acestep.sam_audio_segment.settings.normalize_ranker_mode",
            return_value="none",
        ):
            settings = SamAudioSettings.from_payload({"ranker_mode": "clap"})

        self.assertEqual("none", settings.ranker_mode)


if __name__ == "__main__":
    unittest.main()
