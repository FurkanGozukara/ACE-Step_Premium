"""Tests for conservative raw lyric cleanup."""

import unittest

from acestep.training.dataset_builder_modules.lyrics_cleanup import (
    clean_raw_lyrics_for_training,
)


class LyricsCleanupTests(unittest.TestCase):
    """Coverage for transcript cleanup fallback behavior."""

    def test_clean_raw_lyrics_adds_sections_and_removes_noise_headers(self) -> None:
        """Transcript-like text should become structured training lyrics."""

        cleaned = clean_raw_lyrics_for_training(
            "\n".join(
                [
                    "Transcript",
                    "opening line here",
                    "second line here",
                    "third line here",
                    "channel news out of nowhere",
                ]
            )
        )

        self.assertIn("[Intro]", cleaned)
        self.assertIn("opening line here", cleaned)
        self.assertIn("[Verse 1]", cleaned)
        self.assertNotIn("Transcript", cleaned)
        self.assertNotIn("news out of", cleaned)

    def test_clean_raw_lyrics_collapses_obvious_repetition(self) -> None:
        """Repeated words and phrases should be capped in fallback lyrics."""

        cleaned = clean_raw_lyrics_for_training(
            "so so so so so so so so\nshake it now shake it now shake it now shake it now"
        )

        self.assertIn("so so so so", cleaned)
        self.assertNotIn("so so so so so", cleaned)
        self.assertIn("shake it now shake it now", cleaned)
        self.assertNotIn("shake it now shake it now shake it now", cleaned)


if __name__ == "__main__":
    unittest.main()
