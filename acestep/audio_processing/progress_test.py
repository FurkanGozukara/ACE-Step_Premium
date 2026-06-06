"""Tests for Audio Processing subprocess progress lines."""

from __future__ import annotations

import unittest

from acestep.audio_processing.progress import encode_progress_line, parse_progress_line


class AudioProcessingProgressTests(unittest.TestCase):
    """Verify Audio Processing subprocess progress serialization."""

    def test_round_trips_progress_line(self) -> None:
        """Encoded progress lines should parse back into clamped values."""

        line = encode_progress_line(1.5, "Running Auto-Editor")

        self.assertEqual((1.0, "Running Auto-Editor"), parse_progress_line(line))

    def test_ignores_unrelated_lines(self) -> None:
        """Non-progress worker output should not be treated as progress."""

        self.assertIsNone(parse_progress_line("regular log line"))


if __name__ == "__main__":
    unittest.main()
