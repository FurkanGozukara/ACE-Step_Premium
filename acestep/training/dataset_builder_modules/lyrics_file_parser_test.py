"""Tests for parsing formatted lyric text files."""

from __future__ import annotations

import unittest

from acestep.training.dataset_builder_modules.lyrics_file_parser import parse_lyrics_text_file


class LyricsFileParserTests(unittest.TestCase):
    """Coverage for plain and sectioned lyric sidecar parsing."""

    def test_parse_codex_formatted_lyrics_file(self) -> None:
        """Caption, lyrics, and metadata sections should be separated."""

        parsed = parse_lyrics_text_file(
            "\n".join(
                [
                    "# Caption",
                    "High-energy West Coast hip-hop anthem.",
                    "",
                    "# Lyrics",
                    "[Intro - talkbox hook]",
                    "California love",
                    "",
                    "[Verse 1]",
                    "Now let me welcome everybody to the wild, wild west",
                    "",
                    "# Metadata",
                    "bpm: 92",
                    "keyscale: G minor",
                    "timesignature: 4",
                    "vocal_language: en",
                    "instrumental: false",
                    "",
                    "# Revision Notes",
                    "- Keep this out of lyrics.",
                ]
            )
        )

        self.assertEqual("High-energy West Coast hip-hop anthem.", parsed.caption)
        self.assertIn("[Intro - talkbox hook]", parsed.lyrics)
        self.assertIn("Now let me welcome everybody", parsed.lyrics)
        self.assertNotIn("# Metadata", parsed.lyrics)
        self.assertNotIn("Revision Notes", parsed.lyrics)
        self.assertEqual(92, parsed.metadata["bpm"])
        self.assertEqual("G minor", parsed.metadata["keyscale"])
        self.assertEqual("4", parsed.metadata["timesignature"])
        self.assertEqual("en", parsed.metadata["vocal_language"])
        self.assertFalse(parsed.metadata["instrumental"])

    def test_plain_lyrics_are_preserved(self) -> None:
        """Files without recognized sections should be treated as lyrics only."""

        lyrics = "[Verse]\nplain lyric line\n\n[Chorus]\nplain hook"

        parsed = parse_lyrics_text_file(lyrics)

        self.assertEqual(lyrics, parsed.lyrics)
        self.assertEqual("", parsed.caption)
        self.assertEqual({}, parsed.metadata)


if __name__ == "__main__":
    unittest.main()
