"""Tests for SAM-Audio VRAM preset selection."""

import unittest

from acestep.sam_audio_segment.vram_presets import (
    SAM_VRAM_PRESET_8GB,
    SAM_VRAM_PRESET_10GB,
    SAM_VRAM_PRESET_12GB,
    SAM_VRAM_PRESET_16GB,
    SAM_VRAM_PRESET_24GB,
    SAM_VRAM_PRESET_32GB,
    SAM_VRAM_PRESET_CHOICES,
    get_sam_vram_preset,
    normalize_sam_vram_preset,
    select_sam_vram_preset_for_gpu,
)


class TestSamAudioVramPresets(unittest.TestCase):
    """Verify SAM-Audio preset thresholds and legacy aliases."""

    def test_gpu_thresholds_select_visible_presets(self):
        """Detected VRAM should map to concrete visible preset names."""

        cases = [
            (31.8, SAM_VRAM_PRESET_32GB),
            (24.0, SAM_VRAM_PRESET_24GB),
            (16.0, SAM_VRAM_PRESET_16GB),
            (12.0, SAM_VRAM_PRESET_12GB),
            (10.0, SAM_VRAM_PRESET_10GB),
            (8.0, SAM_VRAM_PRESET_8GB),
        ]
        for memory_gb, expected in cases:
            with self.subTest(memory_gb=memory_gb):
                self.assertEqual(expected, select_sam_vram_preset_for_gpu(memory_gb))

    def test_legacy_aliases_normalize_to_current_presets(self):
        """Older saved custom presets should still load."""

        self.assertEqual(SAM_VRAM_PRESET_16GB, normalize_sam_vram_preset("16gb_low"))
        self.assertEqual(SAM_VRAM_PRESET_16GB, normalize_sam_vram_preset("fp8_scaled"))

    def test_low_vram_presets_enable_lite_mode(self):
        """Lower VRAM presets should reduce optional component memory."""

        self.assertTrue(get_sam_vram_preset(SAM_VRAM_PRESET_10GB)["low_vram_lite"])
        self.assertEqual(
            "auto",
            get_sam_vram_preset(SAM_VRAM_PRESET_10GB)["attention_backend"],
        )
        self.assertEqual("cpu", get_sam_vram_preset(SAM_VRAM_PRESET_8GB)["device_mode"])
        self.assertEqual(
            "auto",
            get_sam_vram_preset(SAM_VRAM_PRESET_8GB)["attention_backend"],
        )

    def test_all_presets_use_paper_default_window(self):
        """All presets should default to the paper long-audio window."""

        for _, preset_name in SAM_VRAM_PRESET_CHOICES:
            with self.subTest(preset_name=preset_name):
                preset = get_sam_vram_preset(preset_name)

                self.assertEqual(20.0, preset["chunk_seconds"])
                self.assertEqual(5.0, preset["chunk_overlap_seconds"])
                self.assertEqual("chunked", preset["long_audio_mode"])


if __name__ == "__main__":
    unittest.main()
