"""Tests for Auto-Editor video command helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.audio_processing.auto_editor_trim_settings import AutoEditorTrimSettings
from acestep.audio_processing.auto_editor_video import (
    _stream_bitrates_from_probe,
    run_auto_editor_video,
)
from acestep.audio_processing.video_reencode_settings import VideoReencodeSettings


class AutoEditorVideoTests(unittest.TestCase):
    """Verify Auto-Editor video command construction."""

    @patch("acestep.audio_processing.auto_editor_video._auto_quality_args")
    @patch("acestep.audio_processing.auto_editor_video.run_command")
    @patch("acestep.audio_processing.auto_editor_video.auto_editor_command")
    def test_auto_quality_uses_probed_bitrate_args(
        self,
        command_mock,
        run_mock,
        quality_mock,
    ) -> None:
        """Auto quality should add probed bitrate args and skip manual codec args."""

        command_mock.return_value = ["auto-editor"]
        quality_mock.return_value = ["--video-bitrate", "6000k", "--audio-bitrate", "192k"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = run_auto_editor_video(
                "input.mp4",
                Path(temp_dir) / "out" / "rendered.mp4",
                AutoEditorTrimSettings(threshold_db=-35.0, margin_seconds=0.2),
                VideoReencodeSettings(auto_set_quality=True, video_codec="libx265"),
            )

        cmd = run_mock.call_args.args[0]
        self.assertTrue(output.endswith("out/rendered.mp4"))
        self.assertIn("--video-bitrate", cmd)
        self.assertIn("6000k", cmd)
        self.assertIn("--audio-bitrate", cmd)
        self.assertIn("192k", cmd)
        self.assertNotIn("--video-codec", cmd)
        self.assertIn("--progress", cmd)
        self.assertIn("ascii", cmd)

    @patch("acestep.audio_processing.auto_editor_video.run_command")
    @patch("acestep.audio_processing.auto_editor_video.auto_editor_command")
    def test_manual_reencode_settings_are_forwarded(self, command_mock, run_mock) -> None:
        """Manual reencode settings should be passed to Auto-Editor."""

        command_mock.return_value = ["auto-editor"]

        run_auto_editor_video(
            "input.mp4",
            "rendered.mp4",
            AutoEditorTrimSettings(threshold_db=-40.0, margin_seconds=0.1),
            VideoReencodeSettings(
                auto_set_quality=False,
                video_codec="libx265",
                video_bitrate="",
                video_crf=20,
                video_preset="slow",
                audio_codec="aac",
                audio_bitrate="256k",
            ),
        )

        cmd = run_mock.call_args.args[0]
        self.assertIn("--video-codec", cmd)
        self.assertIn("libx265", cmd)
        self.assertIn("-crf", cmd)
        self.assertIn("20", cmd)
        self.assertIn("--preset", cmd)
        self.assertIn("slow", cmd)
        self.assertIn("--audio-bitrate", cmd)
        self.assertIn("256k", cmd)
        self.assertIn("--progress", cmd)
        self.assertIn("ascii", cmd)

    def test_stream_bitrates_from_probe_returns_first_audio_video_streams(self) -> None:
        """ffprobe stream bitrates should be formatted in kilobits."""

        payload = {
            "streams": [
                {"codec_type": "video", "bit_rate": "6000000"},
                {"codec_type": "audio", "bit_rate": "192000"},
                {"codec_type": "audio", "bit_rate": "128000"},
            ]
        }

        bitrates = _stream_bitrates_from_probe(json.dumps(payload))

        self.assertEqual({"video": "6000k", "audio": "192k"}, bitrates)


if __name__ == "__main__":
    unittest.main()
