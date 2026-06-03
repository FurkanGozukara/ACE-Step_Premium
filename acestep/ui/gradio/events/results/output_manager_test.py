"""Unit tests for generation output-folder persistence helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.results.output_manager import persist_generation_inputs


class OutputManagerTests(unittest.TestCase):
    """Verify run-folder helpers persist the expected companion files."""

    def test_persist_generation_inputs_writes_text_request_and_copies_audio(self):
        """Caption, lyrics, request JSON, and input audio copies should be saved."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "0001"
            run_dir.mkdir()

            reference_audio = root / "reference.wav"
            reference_audio.write_bytes(b"reference-bytes")
            source_audio = root / "source.mp3"
            source_audio.write_bytes(b"source-bytes")

            assets = persist_generation_inputs(
                run_dir=run_dir,
                caption="A polished synth-pop chorus",
                lyrics="[verse]\nhello world",
                reference_audio=str(reference_audio),
                src_audio=str(source_audio),
                request_payload={
                    "task_type": "text2music",
                    "audio_format": "mp3",
                    "thinking": True,
                },
            )

            caption_path = Path(assets["caption_path"])
            lyrics_path = Path(assets["lyrics_path"])
            request_path = Path(assets["request_path"])
            copied_reference_path = Path(assets["reference_audio_path"])
            copied_source_path = Path(assets["source_audio_path"])

            self.assertEqual(caption_path.read_text(encoding="utf-8"), "A polished synth-pop chorus")
            self.assertEqual(lyrics_path.read_text(encoding="utf-8"), "[verse]\nhello world")
            self.assertEqual(copied_reference_path.read_bytes(), b"reference-bytes")
            self.assertEqual(copied_source_path.read_bytes(), b"source-bytes")

            payload = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["request"]["task_type"], "text2music")
            self.assertEqual(payload["request"]["audio_format"], "mp3")
            self.assertEqual(payload["assets"]["caption_path"], str(caption_path).replace("\\", "/"))
            self.assertEqual(
                payload["assets"]["reference_audio_path"],
                str(copied_reference_path).replace("\\", "/"),
            )
            self.assertEqual(
                payload["assets"]["source_audio_path"],
                str(copied_source_path).replace("\\", "/"),
            )

    def test_persist_generation_inputs_uses_newest_stale_upload_list(self):
        """Persisted generation assets should tolerate stale Gradio upload lists."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "0001"
            run_dir.mkdir()

            old_reference = root / "old_reference.wav"
            old_reference.write_bytes(b"old")
            new_reference = root / "new_reference.wav"
            new_reference.write_bytes(b"new")

            assets = persist_generation_inputs(
                run_dir=run_dir,
                caption="caption",
                lyrics="lyrics",
                reference_audio=[str(old_reference), str(new_reference)],
                src_audio=None,
                request_payload={},
            )

            copied_reference_path = Path(assets["reference_audio_path"])
            self.assertEqual(copied_reference_path.read_bytes(), b"new")
            self.assertEqual(
                assets["original_reference_audio"],
                str(new_reference),
            )


if __name__ == "__main__":
    unittest.main()
