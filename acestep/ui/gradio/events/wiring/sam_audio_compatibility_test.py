"""Tests for SAM-Audio UI compatibility updates."""

import unittest

from acestep.sam_audio_segment.settings import SAM_AUDIO_LONG_MODE_MULTIDIFFUSION
from acestep.ui.gradio.events.wiring.sam_audio_compatibility import (
    apply_sam_audio_compatibility,
)


class TestSamAudioCompatibility(unittest.TestCase):
    """Verify incompatible SAM-Audio UI options are forced to safe values."""

    def test_lite_mode_disables_rankers_spans_and_visual_prompt(self):
        """Lite mode should force settings that the backend can load."""

        result = apply_sam_audio_compatibility("visual", True, "chunked")

        self.assertEqual("text", result[0]["value"])
        self.assertEqual((("Text", "text"), ("Span", "span")), result[0]["choices"])
        self.assertEqual("none", result[1]["value"])
        self.assertFalse(result[1]["interactive"])
        self.assertFalse(result[2]["value"])
        self.assertFalse(result[2]["interactive"])
        self.assertIsNone(result[9]["value"])
        self.assertFalse(result[9]["interactive"])

    def test_multidiffusion_disables_all_non_text_options(self):
        """Multi-diffusion should be locked to text-only one-candidate runs."""

        result = apply_sam_audio_compatibility(
            "span",
            False,
            SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
        )

        self.assertEqual("text", result[0]["value"])
        self.assertFalse(result[0]["interactive"])
        self.assertEqual((("Text", "text"),), result[0]["choices"])
        self.assertEqual("none", result[1]["value"])
        self.assertFalse(result[2]["value"])
        self.assertEqual(1, result[3]["value"])
        self.assertFalse(result[3]["interactive"])
        self.assertFalse(result[4]["value"])
        self.assertFalse(result[4]["interactive"])
        self.assertEqual("", result[5]["value"])
        self.assertFalse(result[5]["interactive"])
        self.assertFalse(result[6]["interactive"])
        self.assertFalse(result[7]["interactive"])
        self.assertFalse(result[8]["interactive"])
        self.assertFalse(result[9]["interactive"])

    def test_standard_mode_enables_rankers_and_visual_when_prompt_is_visual(self):
        """Normal chunked mode should keep optional visual and ranker controls usable."""

        result = apply_sam_audio_compatibility("visual", False, "chunked")

        self.assertEqual("visual", result[0]["value"])
        self.assertTrue(result[0]["interactive"])
        self.assertTrue(result[1]["interactive"])
        self.assertTrue(result[2]["interactive"])
        self.assertTrue(result[3]["interactive"])
        self.assertTrue(result[9]["interactive"])


if __name__ == "__main__":
    unittest.main()
