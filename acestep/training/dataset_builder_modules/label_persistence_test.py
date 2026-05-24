"""Tests for auto-label sidecar persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training.dataset_builder_modules.label_persistence import (
    save_sample_label_metadata,
)
from acestep.training.dataset_builder_modules.models import AudioSample
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class LabelPersistenceTests(unittest.TestCase):
    """Verify labels are saved where scan can find them later."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_save_sample_label_metadata_writes_audio_sidecar(self) -> None:
        """A completed label should be visible in a per-audio JSON file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            audio_path = Path(tmpdir) / "sample.mp3"
            audio_path.write_bytes(b"audio")
            sample = AudioSample(
                audio_path=str(audio_path),
                filename=audio_path.name,
                caption="warm piano ballad",
                genre="piano",
                bpm=90,
                keyscale="C major",
                timesignature="4",
                language="en",
                is_instrumental=False,
                labeled=True,
            )

            sidecar_path = save_sample_label_metadata(sample)

            data = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
            self.assertEqual(str(audio_path), data["audio_path"])
            self.assertEqual("warm piano ballad", data["caption"])
            self.assertEqual("piano", data["genre"])
            self.assertTrue(data["labeled"])

    def test_save_sample_label_metadata_writes_processed_label_folder(self) -> None:
        """Processed labels should be saved outside the original source folder."""

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            label_dir = Path(tmpdir) / "processed_labels"
            source_dir.mkdir()
            audio_path = source_dir / "sample.mp3"
            lyrics_path = source_dir / "sample.txt"
            audio_path.write_bytes(b"audio")
            lyrics_path.write_text("original lyrics", encoding="utf-8")
            set_safe_roots([tmpdir])
            sample = AudioSample(
                audio_path=str(audio_path),
                filename=audio_path.name,
                caption="processed caption",
                caption_source="lyrics_file",
                lyrics="[Verse]\nprocessed lyrics",
                raw_lyrics="original lyrics",
                labeled=True,
            )

            label_path = save_sample_label_metadata(
                sample,
                output_dir=str(label_dir),
                source_root=str(source_dir),
            )

            self.assertEqual(label_dir / "sample.json", Path(label_path))
            self.assertFalse(audio_path.with_suffix(".json").exists())
            self.assertEqual("original lyrics", lyrics_path.read_text(encoding="utf-8"))
            data = json.loads(Path(label_path).read_text(encoding="utf-8"))
            self.assertEqual(str(audio_path), data["audio_path"])
            self.assertEqual("processed caption", data["caption"])
            self.assertEqual("lyrics_file", data["caption_source"])
            self.assertEqual("[Verse]\nprocessed lyrics", data["lyrics"])


if __name__ == "__main__":
    unittest.main()
