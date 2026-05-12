"""Tests for resolving original simple-tab output paths from Gradio values."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.wiring.simple_run_paths import resolve_simple_audio_path


class SimpleRunPathsTests(unittest.TestCase):
    """Verify Gradio cache paths map back to the generated run folder."""

    def test_resolves_audio_from_generation_manifest(self):
        """A cached audio path should resolve through the manifest sample row."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "0011"
            cache_dir = root / "cache"
            run_dir.mkdir()
            cache_dir.mkdir()
            output_audio = run_dir / "song.flac"
            cache_audio = cache_dir / "song.flac"
            manifest_path = run_dir / "generation_manifest.json"
            output_audio.write_bytes(b"audio")
            cache_audio.write_bytes(b"audio")
            manifest_path.write_text(
                json.dumps({"samples": [{"audio_path": _normalized(output_audio)}]}),
                encoding="utf-8",
            )

            resolved = resolve_simple_audio_path(
                str(cache_audio),
                [str(manifest_path), str(output_audio)],
            )

            self.assertEqual(resolved, _normalized(output_audio))

    def test_resolves_audio_from_generated_file_list_without_manifest(self):
        """The generated file list should recover the matching output audio path."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "0011"
            cache_dir = root / "cache"
            run_dir.mkdir()
            cache_dir.mkdir()
            output_audio = run_dir / "song.flac"
            cache_audio = cache_dir / "song.flac"
            output_audio.write_bytes(b"audio")
            cache_audio.write_bytes(b"audio")

            resolved = resolve_simple_audio_path(str(cache_audio), [str(output_audio)])

            self.assertEqual(resolved, _normalized(output_audio))

    def test_returns_input_path_when_no_generated_files_match(self):
        """A path without supporting generated metadata should pass through."""

        self.assertEqual(
            resolve_simple_audio_path("C:/temp/song.flac", []),
            "C:/temp/song.flac",
        )


def _normalized(path: Path) -> str:
    """Return a repository-style normalized absolute path."""

    return str(path.resolve()).replace("\\", "/")


if __name__ == "__main__":
    unittest.main()
