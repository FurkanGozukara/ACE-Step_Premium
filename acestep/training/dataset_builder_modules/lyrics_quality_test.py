"""Tests for training lyric quality checks."""

import unittest

from acestep.training.dataset_builder_modules.lyrics_quality import select_training_lyrics


class LyricsQualityTests(unittest.TestCase):
    """Coverage for selecting safe lyrics after LM formatting."""

    def test_select_training_lyrics_accepts_structured_matching_output(self) -> None:
        """A structured rewrite that preserves words should be accepted."""

        raw_lyrics = "\n".join(
            [
                "moonlight on the avenue",
                "we carry every memory",
                "drums are moving through the room",
                "voices rise in harmony",
            ]
        )
        formatted_lyrics = "\n".join(
            [
                "[Verse 1]",
                "moonlight on the avenue",
                "we carry every memory",
                "",
                "[Chorus]",
                "drums are moving through the room",
                "voices rise in harmony",
            ]
        )

        selection = select_training_lyrics(raw_lyrics, formatted_lyrics)

        self.assertEqual(formatted_lyrics, selection.lyrics)
        self.assertEqual(formatted_lyrics, selection.formatted_lyrics)
        self.assertEqual("", selection.rejection_reason)

    def test_select_training_lyrics_rejects_low_overlap_output(self) -> None:
        """Formatted lyrics should be rejected when they replace the source content."""

        raw_lyrics = "\n".join(
            [
                "silver train beside the river",
                "window light across the floor",
                "old guitar below the staircase",
                "morning rain against the door",
            ]
        )
        formatted_lyrics = "\n".join(
            [
                "[Verse 1]",
                "neon fire above the skyline",
                "dancing fast until the sunrise",
                "[Chorus]",
                "electric hearts are never lonely",
                "take my hand into the spotlight",
            ]
        )

        selection = select_training_lyrics(raw_lyrics, formatted_lyrics)

        self.assertIn("[Intro]", selection.lyrics)
        self.assertIn("silver train beside the river", selection.lyrics)
        self.assertIn("morning rain against the door", selection.lyrics)
        self.assertEqual("", selection.formatted_lyrics)
        self.assertEqual("low overlap with raw lyrics", selection.rejection_reason)

    def test_select_training_lyrics_rejects_single_line_word_loops(self) -> None:
        """A long repeated token line should be rejected even when only one line."""

        raw_lyrics = "west coast rhythm under city lights with voices moving through the room"
        formatted_lyrics = "[Chorus]\n" + " ".join(["so"] * 40)

        selection = select_training_lyrics(raw_lyrics, formatted_lyrics)

        self.assertEqual("", selection.formatted_lyrics)
        self.assertEqual("repetitive formatted lyrics", selection.rejection_reason)

    def test_select_training_lyrics_rejects_transcription_loop_without_raw(self) -> None:
        """Repetitive transcription output should be rejected without raw lyrics."""

        selection = select_training_lyrics("", "[Verse]\n" + " ".join(["yeah"] * 40))

        self.assertEqual("", selection.lyrics)
        self.assertEqual("", selection.formatted_lyrics)
        self.assertEqual("repetitive formatted lyrics", selection.rejection_reason)


if __name__ == "__main__":
    unittest.main()
