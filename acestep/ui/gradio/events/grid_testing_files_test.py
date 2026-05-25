"""Tests for Grid Testing output-file helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.grid_testing_files import flatten_generation_outputs
from acestep.ui.gradio.events.grid_testing_paths import resolve_grid_output_folder


class GridTestingFilesTests(unittest.TestCase):
    """Verify grid output folders and flattened artifacts."""

    def test_empty_output_folder_allocates_next_grid_folder(self) -> None:
        """Default grid runs should create incrementing outputs/grid folders."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs"
            (root / "grid-0001").mkdir(parents=True)
            (root / "grid-0003").mkdir()

            with patch(
                "acestep.ui.gradio.events.grid_testing_paths.DEFAULT_RESULTS_DIR",
                root,
            ):
                target = resolve_grid_output_folder("")

        self.assertEqual("grid-0004", target.name)

    def test_mp3_only_flatten_keeps_only_mp3(self) -> None:
        """MP3-only grid output should skip metadata and text sidecars."""

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            target = Path(tmpdir) / "grid"
            manifest, _metadata, mp3_path = _write_generation_run(run_dir)

            written = flatten_generation_outputs(
                [str(manifest)],
                target,
                prefix="voice",
                caption="style",
                lyrics="lyrics",
                mp3_only=True,
            )

            expected = str((target / "voice-0001.mp3").resolve()).replace("\\", "/")
            self.assertEqual([expected], written)
            self.assertEqual("audio", Path(written[0]).read_text(encoding="utf-8"))
            self.assertFalse((target / "voice-0001.json").exists())
            self.assertTrue(mp3_path.name.endswith(".mp3"))

    def test_full_flatten_writes_audio_metadata_and_text(self) -> None:
        """Full grid output should save prefixed audio, JSON, caption, and lyrics."""

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            target = Path(tmpdir) / "grid"
            manifest, _metadata, _mp3_path = _write_generation_run(run_dir)

            written = flatten_generation_outputs(
                [str(manifest)],
                target,
                prefix="voice",
                caption="style",
                lyrics="lyrics",
                mp3_only=False,
            )

            metadata = json.loads((target / "voice-0001.json").read_text(encoding="utf-8"))

            expected = str((target / "voice-0001.mp3").resolve()).replace("\\", "/")
            self.assertIn(expected, written)
            self.assertEqual("style", (target / "voice-0001_caption.txt").read_text())
            self.assertEqual("lyrics", (target / "voice-0001_lyrics.txt").read_text())
            self.assertEqual(expected, metadata["audio_paths"]["mp3"])
            self.assertEqual(expected, metadata["mp3_path"])
            self.assertEqual(expected, metadata["_meta"]["audio_paths"]["mp3"])
            self.assertEqual(str(target.resolve()).replace("\\", "/"), metadata["_meta"]["run_dir"])
            self.assertEqual("voice-0001", metadata["_meta"]["grid_output_stem"])


def _write_generation_run(run_dir: Path) -> tuple[Path, Path, Path]:
    """Create a minimal generated run manifest for tests."""

    run_dir.mkdir(parents=True)
    mp3_path = run_dir / "0001.mp3"
    metadata_path = run_dir / "0001.json"
    manifest_path = run_dir / "generation_manifest.json"
    mp3_path.write_text("audio", encoding="utf-8")
    metadata_path.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_index": 1,
                        "audio_paths": {"mp3": str(mp3_path)},
                        "mp3_path": str(mp3_path),
                        "metadata_path": str(metadata_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, metadata_path, mp3_path


if __name__ == "__main__":
    unittest.main()
