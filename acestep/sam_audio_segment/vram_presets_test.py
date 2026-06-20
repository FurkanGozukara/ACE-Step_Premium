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
            (28.0, SAM_VRAM_PRESET_32GB),
            (20.0, SAM_VRAM_PRESET_24GB),
            (14.0, SAM_VRAM_PRESET_16GB),
            (11.0, SAM_VRAM_PRESET_12GB),
            (10.0, SAM_VRAM_PRESET_10GB),
            (9.99, SAM_VRAM_PRESET_8GB),
        ]
        for memory_gb, expected in cases:
            with self.subTest(memory_gb=memory_gb):
                self.assertEqual(expected, select_sam_vram_preset_for_gpu(memory_gb))

    def test_legacy_aliases_normalize_to_current_presets(self):
        """Older saved custom presets should still load."""

        self.assertEqual(SAM_VRAM_PRESET_16GB, normalize_sam_vram_preset("16gb_low"))
        self.assertEqual(SAM_VRAM_PRESET_16GB, normalize_sam_vram_preset("fp8_scaled"))

    def test_quality_presets_match_measured_candidate_profiles(self):
        """Measured quality tiers should set candidates and rankers explicitly."""

        highest = get_sam_vram_preset(SAM_VRAM_PRESET_32GB)
        high = get_sam_vram_preset(SAM_VRAM_PRESET_24GB)
        normal = get_sam_vram_preset(SAM_VRAM_PRESET_16GB)

        self.assertEqual(4, highest["reranking_candidates"])
        self.assertEqual("judge", highest["ranker_mode"])
        self.assertFalse(highest["low_vram_lite"])
        self.assertEqual(2, high["reranking_candidates"])
        self.assertEqual("judge", high["ranker_mode"])
        self.assertFalse(high["low_vram_lite"])
        self.assertEqual(1, normal["reranking_candidates"])
        self.assertEqual("none", normal["ranker_mode"])

    def test_low_vram_gpu_presets_use_measured_10s_windows(self):
        """12GB and 10GB presets should use the measured CUDA-safe window."""

        for preset_name in (SAM_VRAM_PRESET_12GB, SAM_VRAM_PRESET_10GB):
            with self.subTest(preset_name=preset_name):
                preset = get_sam_vram_preset(preset_name)

                self.assertTrue(preset["low_vram_lite"])
                self.assertEqual("fp8_scaled", preset["quantization"])
                self.assertEqual(32, preset["ode_steps"])
                self.assertEqual("auto", preset["device_mode"])
                self.assertEqual(10.0, preset["chunk_seconds"])
                self.assertEqual(5.0, preset["chunk_overlap_seconds"])

    def test_8gb_preset_keeps_cpu_safe_lite_mode(self):
        """8GB should stay CPU-safe because the measured 10s CUDA peak is too tight."""

        self.assertEqual("cpu", get_sam_vram_preset(SAM_VRAM_PRESET_8GB)["device_mode"])
        self.assertEqual(
            "auto",
            get_sam_vram_preset(SAM_VRAM_PRESET_8GB)["attention_backend"],
        )

    def test_visible_labels_explain_quality_levels(self):
        """Preset names should explain candidate quality tradeoffs."""

        labels = [label for label, _ in SAM_VRAM_PRESET_CHOICES]

        self.assertIn(
            "32GB Highest Quality - 4 candidates + Judge (peak 24.5GiB)",
            labels,
        )
        self.assertIn(
            "24GB High Quality - 2 candidates + Judge (peak 17.6GiB)",
            labels,
        )
        self.assertIn(
            "12GB Normal Quality - 10s GPU chunks (peak 9.3GiB)",
            labels,
        )
        self.assertIn(
            "10GB Normal Quality - 10s GPU chunks (peak 9.3GiB)",
            labels,
        )
        self.assertTrue(any("Normal Quality" in label for label in labels))

    def test_quality_presets_use_paper_default_window(self):
        """Quality presets should default to the paper long-audio window."""

        for preset_name in (SAM_VRAM_PRESET_32GB, SAM_VRAM_PRESET_24GB, SAM_VRAM_PRESET_16GB):
            with self.subTest(preset_name=preset_name):
                preset = get_sam_vram_preset(preset_name)

                self.assertEqual(20.0, preset["chunk_seconds"])
                self.assertEqual(5.0, preset["chunk_overlap_seconds"])
                self.assertEqual("chunked", preset["long_audio_mode"])

    def test_all_presets_use_chunked_long_audio_mode(self):
        """Every visible preset should keep chunked long-audio processing enabled."""

        for _, preset_name in SAM_VRAM_PRESET_CHOICES:
            with self.subTest(preset_name=preset_name):
                self.assertEqual("chunked", get_sam_vram_preset(preset_name)["long_audio_mode"])


if __name__ == "__main__":
    unittest.main()
