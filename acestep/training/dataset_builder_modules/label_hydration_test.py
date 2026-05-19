"""Tests for hydrating scanned samples from processed label folders."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training.dataset_builder_modules.label_hydration import (
    has_unlabeled_samples,
    hydrate_samples_from_label_dir,
)
from acestep.training.dataset_builder_modules.models import AudioSample
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class LabelHydrationTests(unittest.TestCase):
    """Verify processed labels are reused for only-unlabeled auto-label runs."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_hydrates_matching_sample_from_processed_label_folder(self) -> None:
        """A matching processed label should mark the scanned sample labeled."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            root = Path(tmpdir)
            audio_path = root / "audio" / "song.flac"
            label_dir = root / "auto_label"
            audio_path.parent.mkdir()
            label_dir.mkdir()
            audio_path.write_bytes(b"audio")
            _write_label(label_dir / "song.json", audio_path, "cached caption")
            sample = AudioSample(audio_path=str(audio_path), filename=audio_path.name)

            hydrated = hydrate_samples_from_label_dir([sample], str(label_dir))

            self.assertEqual(1, hydrated)
            self.assertTrue(sample.labeled)
            self.assertFalse(has_unlabeled_samples([sample]))
            self.assertEqual("cached caption", sample.caption)
            self.assertEqual("hip hop", sample.genre)
            self.assertEqual("en", sample.language)
            self.assertFalse(sample.is_instrumental)

    def test_ignores_non_matching_processed_label(self) -> None:
        """Labels for other audio paths should not affect scanned samples."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            root = Path(tmpdir)
            audio_path = root / "audio" / "song.flac"
            other_audio = root / "audio" / "other.flac"
            label_dir = root / "auto_label"
            audio_path.parent.mkdir()
            label_dir.mkdir()
            audio_path.write_bytes(b"audio")
            other_audio.write_bytes(b"audio")
            _write_label(label_dir / "other.json", other_audio, "other caption")
            sample = AudioSample(audio_path=str(audio_path), filename=audio_path.name)

            hydrated = hydrate_samples_from_label_dir([sample], str(label_dir))

            self.assertEqual(0, hydrated)
            self.assertTrue(has_unlabeled_samples([sample]))
            self.assertFalse(sample.labeled)


def _write_label(label_path: Path, audio_path: Path, caption: str) -> None:
    """Write a processed-label JSON file."""

    label_path.write_text(
        json.dumps(
            {
                "audio_path": str(audio_path),
                "filename": audio_path.name,
                "caption": caption,
                "genres": "hip hop",
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
