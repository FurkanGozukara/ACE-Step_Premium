"""Tests for shared media IO helpers."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.audio_processing.media_duration import FFPROBE_TIMEOUT_SECONDS
from acestep.audio_processing.media_io import media_audio_duration_seconds


class MediaIoDurationTests(unittest.TestCase):
    """Verify duration probing avoids full media decode for upload-time UI work."""

    @patch("acestep.audio_processing.media_io._read_with_ffmpeg")
    @patch("acestep.audio_processing.media_duration.subprocess.run")
    def test_video_duration_uses_ffprobe_audio_stream_metadata(
        self,
        run_mock,
        read_mock,
    ) -> None:
        """Video duration should come from ffprobe metadata, not full audio decode."""

        run_mock.return_value = SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "duration": "90.0"},
                        {"codec_type": "audio", "duration": "12.5"},
                    ],
                    "format": {"duration": "99.0"},
                }
            ),
            stderr="",
        )

        duration = media_audio_duration_seconds("clip.mp4")

        self.assertEqual(duration, 12.5)
        read_mock.assert_not_called()
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "ffprobe")
        self.assertEqual(run_mock.call_args.kwargs["timeout"], FFPROBE_TIMEOUT_SECONDS)

    @patch("acestep.audio_processing.media_duration.subprocess.run")
    def test_video_duration_falls_back_to_format_metadata(self, run_mock) -> None:
        """Format duration should be used when audio stream duration is absent."""

        run_mock.return_value = SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "audio", "duration": "N/A"}],
                    "format": {"duration": "8.75"},
                }
            ),
            stderr="",
        )

        duration = media_audio_duration_seconds("clip.webm")

        self.assertEqual(duration, 8.75)


if __name__ == "__main__":
    unittest.main()
