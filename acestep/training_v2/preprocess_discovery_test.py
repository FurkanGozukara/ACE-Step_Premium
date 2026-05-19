"""Tests for cross-platform audio discovery in the V2 preprocessing pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.training_v2.preprocess import preprocess_audio_files
from acestep.training_v2.preprocess_discovery import (
    discover_audio_files,
    load_sample_metadata,
)


class _CpuGpu:
    """Minimal GPU-detection result for preprocessing orchestration tests."""

    device = "cpu"
    precision = "float32"


class PreprocessDiscoveryPathTests(unittest.TestCase):
    """Verify audio discovery handles Windows- and Linux-friendly filenames."""

    def test_discovers_special_filenames_recursively(self) -> None:
        """Directory discovery should preserve punctuation-heavy audio filenames."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "disc 1"
            nested.mkdir()
            filenames = [
                "02 - Hit 'Em Up - Single Version.flac",
                "04 - All Eyez On Me (ft. Big Syke).flac",
                "22 - To Live & Die In L.A..flac",
                "Run Tha Streetz (ft. Michel'le, Storm, Mutah).flac",
            ]
            for filename in filenames[:3]:
                (root / filename).write_bytes(b"audio")
            (nested / filenames[3]).write_bytes(b"audio")

            discovered = discover_audio_files(tmpdir, None)

        self.assertEqual(sorted(filenames), sorted(path.name for path in discovered))

    def test_dataset_json_resolves_windows_separator_relative_paths(self) -> None:
        """JSON paths with backslashes should resolve on either host OS."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            audio_path = nested / "02 - Hit 'Em Up - Single Version.flac"
            audio_path.write_bytes(b"audio")
            dataset_json = root / "dataset.json"
            dataset_json.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "audio_path": r"nested\02 - Hit 'Em Up - Single Version.flac",
                                "filename": r"nested\02 - Hit 'Em Up - Single Version.flac",
                                "caption": "west coast hip hop",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            discovered = discover_audio_files(None, str(dataset_json))
            metadata = load_sample_metadata(str(dataset_json), discovered)

        self.assertEqual([audio_path], discovered)
        self.assertEqual("west coast hip hop", metadata[audio_path.name]["caption"])

    def test_preprocess_pipeline_accepts_special_filenames(self) -> None:
        """The public preprocessing entrypoint should pass special names through."""

        captured_names: list[str] = []

        def fake_pass1(**kwargs):
            """Capture discovered audio files without loading model checkpoints."""
            captured_names.extend(path.name for path in kwargs["audio_files"])
            return [], 0

        def fake_pass2(**kwargs):
            """Return an empty successful heavy-pass result."""
            return 0, 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "song_dataset_final"
            source.mkdir()
            (source / "04 - All Eyez On Me (ft. Big Syke).flac").write_bytes(b"audio")
            (source / "23 - Picture Me Rollin'.flac").write_bytes(b"audio")
            output = root / "tensors"

            with (
                patch("acestep.training_v2.gpu_utils.detect_gpu", return_value=_CpuGpu()),
                patch("acestep.training_v2.preprocess._pass1_light", side_effect=fake_pass1),
                patch("acestep.training_v2.preprocess._pass2_heavy", side_effect=fake_pass2),
            ):
                result = preprocess_audio_files(
                    audio_dir=str(source),
                    output_dir=str(output),
                    checkpoint_dir=str(root / "checkpoints"),
                    device="cpu",
                    precision="float32",
                )

        self.assertEqual(2, result["total"])
        self.assertIn("04 - All Eyez On Me (ft. Big Syke).flac", captured_names)
        self.assertIn("23 - Picture Me Rollin'.flac", captured_names)


if __name__ == "__main__":
    unittest.main()
