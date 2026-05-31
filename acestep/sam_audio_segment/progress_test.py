"""Tests for SAM-Audio progress helpers."""

import unittest

from acestep.sam_audio_segment.progress import (
    encode_progress_line,
    parse_progress_line,
    report_progress,
)


class TestSamAudioProgress(unittest.TestCase):
    """Verify progress clamping and subprocess line parsing."""

    def test_report_progress_clamps_fraction(self):
        """Progress callbacks should receive normalized fractions."""

        events: list[tuple[float, str]] = []

        report_progress(
            lambda fraction, message: events.append((fraction, message)),
            1.5,
            "done",
        )

        self.assertEqual([(1.0, "done")], events)

    def test_progress_line_round_trip(self):
        """Subprocess progress lines should decode into progress events."""

        line = encode_progress_line(0.25, "Loading")

        self.assertEqual((0.25, "Loading"), parse_progress_line(line))

    def test_non_progress_line_is_ignored(self):
        """Regular stdout should not be treated as progress."""

        self.assertIsNone(parse_progress_line("ordinary output"))


if __name__ == "__main__":
    unittest.main()
