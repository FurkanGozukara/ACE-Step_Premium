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


if __name__ == "__main__":
    unittest.main()
