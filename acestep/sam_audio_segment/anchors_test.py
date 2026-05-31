"""Tests for SAM-Audio span anchor parsing."""

import unittest

from acestep.sam_audio_segment.anchors import anchors_for_settings
from acestep.sam_audio_segment.settings import SamAudioSettings


class TestSamAudioAnchors(unittest.TestCase):
    """Verify explicit SAM-Audio span prompt anchors."""

    def test_simple_anchor_from_controls(self):
        """Span mode creates one batch-wrapped anchor from control values."""

        settings = SamAudioSettings(
            prompt_mode="span",
            anchor_polarity="+",
            anchor_start=1.25,
            anchor_end=2.5,
        )

        self.assertEqual([[("+", 1.25, 2.5)]], anchors_for_settings(settings))

    def test_anchor_json_supports_multiple_polarities(self):
        """Anchor JSON supports multiple positive and negative spans."""

        settings = SamAudioSettings(
            use_span_anchor=True,
            anchor_json='[["+", 2.0, 3.5], ["-", 0.0, 1.0]]',
        )

        self.assertEqual(
            [[("+", 2.0, 3.5), ("-", 0.0, 1.0)]],
            anchors_for_settings(settings),
        )


if __name__ == "__main__":
    unittest.main()
