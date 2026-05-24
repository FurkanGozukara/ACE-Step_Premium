"""Tests for dataset directory scanning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.training.dataset_builder import DatasetBuilder
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class ScanMixinTests(unittest.TestCase):
    """Verify scan hydration from existing label sidecars."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_scan_directory_loads_label_sidecar_metadata(self) -> None:
        """Previously saved labels should mark samples labeled on the next scan."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            audio_path = Path(tmpdir) / "song.mp3"
            audio_path.write_bytes(b"audio")
            audio_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "caption": "bright synth pop",
                        "genre": "synth pop",
                        "lyrics": "la la",
                        "bpm": 124,
                        "keyscale": "A minor",
                        "timesignature": "4",
                        "language": "en",
                        "is_instrumental": False,
                        "labeled": True,
                    }
                ),
                encoding="utf-8",
            )
            builder = DatasetBuilder()

            with patch(
                "acestep.training.dataset_builder_modules.scan.get_audio_duration",
                return_value=30,
            ):
                samples, _status = builder.scan_directory(tmpdir)

            self.assertEqual(1, len(samples))
            self.assertTrue(samples[0].labeled)
            self.assertEqual("bright synth pop", samples[0].caption)
            self.assertEqual("synth pop", samples[0].genre)
            self.assertFalse(samples[0].is_instrumental)

    def test_scan_directory_loads_ace_generation_sidecar_aliases(self) -> None:
        """ACE generation JSON sidecars should hydrate vocal language and instrumental fields."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            audio_path = Path(tmpdir) / "generated.mp3"
            audio_path.write_bytes(b"audio")
            audio_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "caption": "known generated caption",
                        "lyrics": "[Verse]\nknown lyric",
                        "vocal_language": "en",
                        "instrumental": False,
                        "bpm": 81,
                        "keyscale": "C major",
                        "timesignature": "4",
                    }
                ),
                encoding="utf-8",
            )
            builder = DatasetBuilder()

            with patch(
                "acestep.training.dataset_builder_modules.scan.get_audio_duration",
                return_value=200,
            ):
                samples, _status = builder.scan_directory(tmpdir)

            self.assertEqual(1, len(samples))
            self.assertEqual("en", samples[0].language)
            self.assertFalse(samples[0].is_instrumental)
            self.assertEqual("[Verse]\nknown lyric", samples[0].lyrics)
            self.assertEqual(81, samples[0].bpm)

    def test_scan_directory_parses_codex_formatted_lyrics_subdir(self) -> None:
        """Formatted lyric files in a sibling subfolder should hydrate lyrics only."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            audio_path = Path(tmpdir) / "01 - Song.flac"
            audio_path.write_bytes(b"audio")
            lyrics_dir = Path(tmpdir) / "codex_formatted_lyrics"
            lyrics_dir.mkdir()
            (lyrics_dir / "01 - Song.txt").write_text(
                "\n".join(
                    [
                        "# Caption",
                        "Bright funk hip-hop with a talkbox hook.",
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
                        "- This should not become training lyrics.",
                    ]
                ),
                encoding="utf-8",
            )
            builder = DatasetBuilder()

            with patch(
                "acestep.training.dataset_builder_modules.scan.get_audio_duration",
                return_value=238,
            ):
                samples, status = builder.scan_directory(tmpdir)

            self.assertIn("Detected 1 lyrics", status)
            self.assertEqual(1, len(samples))
            sample = samples[0]
            self.assertEqual("Bright funk hip-hop with a talkbox hook.", sample.caption)
            self.assertEqual("lyrics_file", sample.caption_source)
            self.assertIn("[Intro - talkbox hook]", sample.raw_lyrics)
            self.assertIn("Now let me welcome everybody", sample.lyrics)
            self.assertNotIn("# Caption", sample.lyrics)
            self.assertNotIn("# Metadata", sample.lyrics)
            self.assertNotIn("Revision Notes", sample.lyrics)
            self.assertEqual(92, sample.bpm)
            self.assertEqual("G minor", sample.keyscale)
            self.assertEqual("4", sample.timesignature)
            self.assertEqual("en", sample.language)
            self.assertFalse(sample.is_instrumental)


if __name__ == "__main__":
    unittest.main()
