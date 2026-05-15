"""Tests for persisted simple-tab MP4 artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.simple_video_artifacts import (
    export_simple_video_artifacts,
)


class SimpleVideoArtifactsTests(unittest.TestCase):
    """Verify MP4 export records the video and source image in run metadata."""

    def test_export_saves_image_mp4_and_metadata(self):
        """The uploaded image and generated MP4 should persist beside the audio."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "0009"
            upload_dir = root / "upload"
            run_dir.mkdir()
            upload_dir.mkdir()
            audio_path = run_dir / "song.flac"
            image_path = upload_dir / "cover.png"
            sidecar_path = run_dir / "song.json"
            audio_path.write_bytes(b"audio")
            image_path.write_bytes(b"image")
            sidecar_path.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
            self._write_run_metadata(run_dir, audio_path, sidecar_path)

            with patch(
                "acestep.ui.gradio.events.wiring.simple_video_artifacts."
                "create_still_image_video",
                side_effect=self._fake_create_video,
            ):
                artifacts = export_simple_video_artifacts(
                    str(audio_path),
                    str(image_path),
                    "720p",
                )

            expected_video = _normalized(run_dir / "song_720p.mp4")
            expected_image = _normalized(run_dir / "video_image.png")
            self.assertEqual(artifacts.video_path, expected_video)
            self.assertEqual(artifacts.image_path, expected_image)
            self.assertEqual((run_dir / "video_image.png").read_bytes(), b"image")
            self.assertTrue((run_dir / "song_720p.mp4").is_file())

            manifest = _read_json(run_dir / "generation_manifest.json")
            sample = manifest["samples"][0]
            self.assertEqual(sample["video_path"], expected_video)
            self.assertEqual(sample["video_image_path"], expected_image)
            self.assertEqual(manifest["request"]["video_path"], expected_video)
            self.assertEqual(manifest["request"]["video_paths"], [expected_video])

            request = _read_json(run_dir / "generation_request.json")
            self.assertEqual(request["request"]["video_path"], expected_video)
            self.assertEqual(request["request"]["video_paths"], [expected_video])
            self.assertEqual(request["assets"]["video_image_path"], expected_image)
            self.assertEqual(request["assets"]["video_paths"], [expected_video])

            sidecar = _read_json(sidecar_path)
            self.assertEqual(sidecar["_meta"]["video_path"], expected_video)
            self.assertEqual(sidecar["_meta"]["video_image_path"], expected_image)

    def _write_run_metadata(
        self,
        run_dir: Path,
        audio_path: Path,
        sidecar_path: Path,
    ) -> None:
        """Write minimal manifest and request metadata for export patching."""

        manifest = {
            "request": {},
            "samples": [
                {
                    "audio_path": _normalized(audio_path),
                    "metadata_path": _normalized(sidecar_path),
                }
            ],
        }
        (run_dir / "generation_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (run_dir / "generation_request.json").write_text(
            json.dumps({"request": {}, "assets": {}}),
            encoding="utf-8",
        )

    def _fake_create_video(
        self,
        *,
        image_path: str,
        audio_path: str,
        output_path: str,
        resolution: str,
    ) -> str:
        """Write a placeholder MP4 file and return the normalized target path."""

        self.assertTrue(Path(image_path).is_file())
        self.assertTrue(Path(audio_path).is_file())
        self.assertEqual(resolution, "720p")
        target = Path(output_path)
        target.write_bytes(b"mp4")
        return _normalized(target)


def _read_json(path: Path) -> dict:
    """Read a JSON test fixture from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized(path: Path) -> str:
    """Return a repository-style normalized absolute path."""

    return str(path.resolve()).replace("\\", "/")


if __name__ == "__main__":
    unittest.main()
