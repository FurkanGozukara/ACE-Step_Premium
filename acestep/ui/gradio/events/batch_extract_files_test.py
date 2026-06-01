"""Tests for Batch Extract filesystem helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.batch_extract_files import copy_batch_extract_audio_outputs


class BatchExtractFilesTests(unittest.TestCase):
    """Verify Batch Extract output-copy filtering."""

    def test_copy_stops_before_source_assets_after_manifest(self) -> None:
        """Only generated sample audio before the manifest should be copied."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated_dir = root / "generated"
            output_dir = root / "output"
            generated_dir.mkdir()
            output_dir.mkdir()
            source_audio = root / "Alpha.wav"
            generated_audio = generated_dir / "generated.flac"
            manifest = generated_dir / "generation_manifest.json"
            source_asset = generated_dir / "source_audio.mp3"
            generated_audio.write_bytes(b"extracted")
            manifest.write_text("{}", encoding="utf-8")
            source_asset.write_bytes(b"original-source")

            copied = copy_batch_extract_audio_outputs(
                [str(generated_audio), str(manifest), str(source_asset)],
                source_audio,
                output_dir,
            )

            self.assertEqual([str(output_dir / "Alpha.flac")], copied)
            self.assertEqual(b"extracted", (output_dir / "Alpha.flac").read_bytes())
            self.assertFalse((output_dir / "Alpha.mp3").exists())


if __name__ == "__main__":
    unittest.main()
