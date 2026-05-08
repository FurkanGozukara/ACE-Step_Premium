"""Tests for generated-song library scanning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.generated_library import scan_generated_songs, select_library_item


class GeneratedLibraryTests(unittest.TestCase):
    """Verify generated-song metadata is read from manifests and sidecars."""

    def test_scans_manifest_samples_with_full_metadata(self) -> None:
        """It returns a selectable record with audio, lyrics, and metadata."""

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "0001"
            run_dir.mkdir()
            audio_path = run_dir / "song.flac"
            audio_path.write_bytes(b"audio")
            sidecar_path = run_dir / "song.json"
            sidecar_path.write_text(
                json.dumps({"lyrics": "sidecar lyrics", "audio_codes": "abc"}),
                encoding="utf-8",
            )
            manifest_path = run_dir / "generation_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "_meta": {"finished_at_utc": "2026-05-08T01:02:03+00:00"},
                        "audio_format": "flac",
                        "request": {
                            "caption": "cinematic synth pop",
                            "lyrics": "request lyrics",
                            "audio_duration": 12,
                            "runtime": {
                                "dit_quantization": "fp8_scaled",
                                "dit_last_init_params": {"config_path": "ace-xl"},
                            },
                        },
                        "samples": [
                            {
                                "sample_index": 1,
                                "key": "song",
                                "audio_path": str(audio_path),
                                "metadata_path": str(sidecar_path),
                                "audio_format": "flac",
                                "score": "Done!",
                                "params": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = scan_generated_songs(tmp)
            audio, details, lyrics, metadata = select_library_item(records[0]["id"], records)

        self.assertEqual(1, len(records))
        self.assertEqual(str(audio_path), audio)
        self.assertIn("ace-xl / fp8_scaled", details)
        self.assertEqual("request lyrics", lyrics)
        self.assertEqual("abc", metadata["audio_codes"])


if __name__ == "__main__":
    unittest.main()
