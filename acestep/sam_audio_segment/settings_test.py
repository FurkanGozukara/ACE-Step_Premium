"""Tests for SAM-Audio settings parsing."""

import unittest

from acestep.sam_audio_segment.settings import SAM_AUDIO_PRESET_KEYS, settings_from_ui_values


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
        self.assertEqual("vocals", settings.effective_prompt)

    def test_values_follow_schema_order(self):
        """Ordered UI values map to the expected settings fields."""

        values = {key: None for key in SAM_AUDIO_PRESET_KEYS}
        values.update(
            {
                "sam_auto_postprocess": True,
                "sam_preserve_original": True,
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
        self.assertTrue(settings.batch_recursive)


if __name__ == "__main__":
    unittest.main()
