"""Tests for batch process filesystem helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.batch_extract_files import copy_batch_extract_audio_outputs


class BatchExtractFilesTests(unittest.TestCase):
    """Verify batch process output-copy filtering."""

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
            remaining_audio = generated_dir / "generated_remaining.flac"
            manifest = generated_dir / "generation_manifest.json"
            source_asset = generated_dir / "source_audio.mp3"
            generated_audio.write_bytes(b"extracted")
            remaining_audio.write_bytes(b"remaining")
            manifest.write_text("{}", encoding="utf-8")
            source_asset.write_bytes(b"original-source")

            copied = copy_batch_extract_audio_outputs(
                [str(generated_audio), str(remaining_audio), str(manifest), str(source_asset)],
                source_audio,
                output_dir,
            )

            self.assertEqual(
                [str(output_dir / "Alpha.flac"), str(output_dir / "Alpha_remaining.flac")],
                copied,
            )
            self.assertEqual(b"extracted", (output_dir / "Alpha.flac").read_bytes())
            self.assertEqual(
                b"remaining",
                (output_dir / "Alpha_remaining.flac").read_bytes(),
            )
            self.assertFalse((output_dir / "Alpha.mp3").exists())

    def test_copy_adds_stem_suffix_for_extract_all_stems(self) -> None:
        """All-stems copies include the requested stem suffix."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated_dir = root / "generated"
            output_dir = root / "output"
            generated_dir.mkdir()
            output_dir.mkdir()
            source_audio = root / "Alpha.wav"
            generated_audio = generated_dir / "generated.flac"
            remaining_audio = generated_dir / "generated_remaining.flac"
            generated_audio.write_bytes(b"extracted")
            remaining_audio.write_bytes(b"remaining")

            copied = copy_batch_extract_audio_outputs(
                [str(generated_audio), str(remaining_audio)],
                source_audio,
                output_dir,
                track_name="vocals",
            )

            self.assertEqual(
                [
                    str(output_dir / "Alpha_vocal.flac"),
                    str(output_dir / "Alpha_vocal_remaining.flac"),
                ],
                copied,
            )

    def test_copy_adds_middle_stem_suffix(self) -> None:
        """Instrument stems use their exact canonical suffix."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated_dir = root / "generated"
            output_dir = root / "output"
            generated_dir.mkdir()
            output_dir.mkdir()
            source_audio = root / "Alpha Song.wav"
            generated_audio = generated_dir / "generated.wav"
            generated_audio.write_bytes(b"guitar")

            copied = copy_batch_extract_audio_outputs(
                [str(generated_audio)],
                source_audio,
                output_dir,
                track_name="guitar",
            )

            self.assertEqual([str(output_dir / "Alpha Song_guitar.wav")], copied)

    def test_copy_avoids_overwriting_source_folder_audio(self) -> None:
        """Batch copy should not overwrite an existing source file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated_dir = root / "generated"
            output_dir = root / "input"
            generated_dir.mkdir()
            output_dir.mkdir()
            source_audio = output_dir / "Alpha.wav"
            generated_audio = generated_dir / "generated.wav"
            source_audio.write_bytes(b"source")
            generated_audio.write_bytes(b"extracted")

            copied = copy_batch_extract_audio_outputs(
                [str(generated_audio)],
                source_audio,
                output_dir,
            )

            self.assertEqual([str(output_dir / "Alpha_extract1.wav")], copied)
            self.assertEqual(b"source", source_audio.read_bytes())
            self.assertEqual(b"extracted", (output_dir / "Alpha_extract1.wav").read_bytes())


if __name__ == "__main__":
    unittest.main()
