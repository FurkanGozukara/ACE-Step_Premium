"""Tests for SAM-Audio optional ranker availability."""

import unittest
from unittest.mock import patch

from acestep.sam_audio_segment.ranker_availability import (
    available_ranker_choices,
    normalize_ranker_mode,
)


class TestRankerAvailability(unittest.TestCase):
    """Verify optional rankers are hidden when their dependencies are absent."""

    def test_missing_clap_disables_clap_and_ensemble(self):
        """CLAP-based modes require the optional laion_clap package."""

        with patch(
            "acestep.sam_audio_segment.ranker_availability.find_spec",
            side_effect=lambda name: (
                object() if name in {"safetensors", "transformers"} else None
            ),
        ), patch(
            "acestep.sam_audio_segment.ranker_availability.local_judge_assets_available",
            return_value=True,
        ):
            choices = dict(available_ranker_choices())

            self.assertIn("none", choices.values())
            self.assertIn("judge", choices.values())
            self.assertNotIn("clap", choices.values())
            self.assertNotIn("text_ensemble", choices.values())
            self.assertEqual("none", normalize_ranker_mode("clap"))

    def test_clap_available_enables_clap_and_ensemble(self):
        """When dependencies exist, CLAP-based rankers should remain selectable."""

        with patch(
            "acestep.sam_audio_segment.ranker_availability.find_spec",
            return_value=object(),
        ), patch(
            "acestep.sam_audio_segment.ranker_availability.local_judge_assets_available",
            return_value=True,
        ):
            choices = dict(available_ranker_choices())

            self.assertIn("clap", choices.values())
            self.assertIn("text_ensemble", choices.values())
            self.assertEqual("clap", normalize_ranker_mode("clap"))

    def test_missing_local_judge_assets_disables_judge(self):
        """Judge requires the local safetensors and processor files."""

        with patch(
            "acestep.sam_audio_segment.ranker_availability.find_spec",
            return_value=object(),
        ), patch(
            "acestep.sam_audio_segment.ranker_availability.local_judge_assets_available",
            return_value=False,
        ):
            choices = dict(available_ranker_choices())

            self.assertNotIn("judge", choices.values())
            self.assertEqual("none", normalize_ranker_mode("judge"))


if __name__ == "__main__":
    unittest.main()
