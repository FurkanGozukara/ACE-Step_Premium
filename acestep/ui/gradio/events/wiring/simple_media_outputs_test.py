"""Tests for simple Create tab media preview helpers."""

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.simple_media_outputs import (
    build_simple_media_preview,
)


class SimpleMediaOutputsTests(unittest.TestCase):
    """Verify audio/video visibility behavior for the simple preview."""

    def test_preview_shows_audio_when_no_image_is_uploaded(self):
        """Audio should remain visible when no video image is provided."""

        updates = list(build_simple_media_preview("song.flac", "done", None, "1080p"))

        self.assertEqual(len(updates), 1)
        audio_update, video_update, status = updates[0]
        self.assertEqual(audio_update["value"], "song.flac")
        self.assertTrue(audio_update["visible"])
        self.assertFalse(video_update["visible"])
        self.assertIn("Audio ready", status)
        self.assertIn("Outputs are saved", status)
        self.assertIn("Folder:", status)

    def test_preview_yields_video_status_and_final_video(self):
        """Image uploads should hide audio and show the generated MP4."""

        with patch(
            "acestep.ui.gradio.events.wiring.simple_media_outputs._create_video",
            return_value="song.mp4",
        ):
            updates = list(
                build_simple_media_preview("song.flac", "done", "cover.png", "720p")
            )

        self.assertEqual(len(updates), 2)
        self.assertIn("Creating MP4", updates[0][2])
        audio_update, video_update, status = updates[1]
        self.assertFalse(audio_update["visible"])
        self.assertEqual(video_update["value"], "song.mp4")
        self.assertTrue(video_update["visible"])
        self.assertIn("MP4 video ready", status)
        self.assertIn("Outputs are saved", status)


if __name__ == "__main__":
    unittest.main()
