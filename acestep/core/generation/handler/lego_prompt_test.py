"""Tests for Lego prompt normalization helpers."""

import unittest

from acestep.core.generation.handler.lego_prompt import normalize_lego_lyrics


class LegoPromptTests(unittest.TestCase):
    """Verify Lego lyrics only condition vocal targets."""

    def test_non_vocal_lego_track_uses_instrumental_lyrics(self):
        """Guitar Lego should not pass full song lyrics to the lyric branch."""

        lyrics = "[Verse]\nSing these words"
        result = normalize_lego_lyrics(
            "lego",
            "Generate the GUITAR track based on the audio context:",
            lyrics,
        )
        self.assertEqual("[Instrumental]", result)

    def test_vocal_lego_track_preserves_lyrics(self):
        """Vocal Lego needs lyric text for sung output."""

        lyrics = "[Verse]\nSing these words"
        result = normalize_lego_lyrics(
            "lego",
            "Generate the VOCALS track based on the audio context:",
            lyrics,
        )
        self.assertEqual(lyrics, result)

    def test_non_lego_task_preserves_lyrics(self):
        """Other tasks keep their existing lyric behavior."""

        lyrics = "[Verse]\nSing these words"
        result = normalize_lego_lyrics(
            "repaint",
            "Repaint the mask area based on the given conditions:",
            lyrics,
        )
        self.assertEqual(lyrics, result)


if __name__ == "__main__":
    unittest.main()
