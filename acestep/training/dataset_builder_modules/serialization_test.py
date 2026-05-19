"""Tests for dataset serialization and processed-label loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training.dataset_builder import DatasetBuilder
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class SerializationMixinTests(unittest.TestCase):
    """Verify dataset loading supports processed auto-label JSON files."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_load_processed_label_directory_uses_audio_path(self) -> None:
        """A folder of processed labels should load samples from full audio paths."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            root = Path(tmpdir)
            audio_dir = root / "audio"
            label_dir = root / "auto_label"
            audio_dir.mkdir()
            label_dir.mkdir()
            first_audio = audio_dir / "01 - Song.flac"
            second_audio = audio_dir / "02 - Song.flac"
            first_audio.write_bytes(b"audio")
            second_audio.write_bytes(b"audio")
            _write_label(label_dir / "01 - Song.json", first_audio, "first caption")
            _write_label(label_dir / "02 - Song.json", second_audio, "second caption")

            builder = DatasetBuilder()
            samples, status = builder.load_dataset(str(label_dir))

            self.assertIn("processed labels", status)
            self.assertEqual(2, len(samples))
            self.assertEqual(str(first_audio), samples[0].audio_path)
            self.assertEqual(str(second_audio), samples[1].audio_path)
            self.assertEqual("first caption", samples[0].caption)
            self.assertTrue(samples[0].labeled)
            self.assertEqual("auto_label", builder.metadata.name)
            self.assertEqual(2, builder.metadata.num_samples)
            self.assertFalse(builder.metadata.all_instrumental)

    def test_load_single_processed_label_file_uses_audio_path(self) -> None:
        """One processed-label JSON should load as a one-sample dataset."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            root = Path(tmpdir)
            audio_path = root / "song.flac"
            label_path = root / "song.json"
            audio_path.write_bytes(b"audio")
            _write_label(label_path, audio_path, "single caption")

            builder = DatasetBuilder()
            samples, status = builder.load_dataset(str(label_path))

            self.assertIn("processed label", status)
            self.assertEqual(1, len(samples))
            self.assertEqual(str(audio_path), samples[0].audio_path)
            self.assertEqual("song.flac", samples[0].filename)
            self.assertEqual("single caption", samples[0].caption)


def _write_label(label_path: Path, audio_path: Path, caption: str) -> None:
    """Write a minimal processed-label JSON file."""

    label_path.write_text(
        json.dumps(
            {
                "audio_path": str(audio_path),
                "filename": audio_path.name,
                "caption": caption,
                "genre": "hip hop",
                "lyrics": "[Verse]\nwords",
                "language": "en",
                "is_instrumental": False,
                "labeled": True,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
