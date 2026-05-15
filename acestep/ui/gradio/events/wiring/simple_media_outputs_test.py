"""Tests for simple Create tab media preview helpers."""

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.simple_media_outputs import (
    build_simple_generated_files_update,
    build_simple_media_preview,
)
from acestep.ui.gradio.events.wiring.simple_video_artifacts import SimpleVideoArtifacts


class SimpleMediaOutputsTests(unittest.TestCase):
    """Verify audio/video visibility behavior for the simple preview."""

    def test_preview_shows_audio_when_no_image_is_uploaded(self):
        """Audio should remain visible when no video image is provided."""

        updates = list(build_simple_media_preview("song.flac", "done", None, "1080p"))

        self.assertEqual(len(updates), 1)
        audio_update, video_update, status, files_update = updates[0]
        self.assertEqual(audio_update["value"], "song.flac")
        self.assertTrue(audio_update["visible"])
        self.assertFalse(video_update["visible"])
        self.assertIn("Audio ready", status)
        self.assertIn("Outputs are saved", status)
        self.assertIn("Folder:", status)
        self.assertFalse(files_update["visible"])

    def test_preview_yields_video_status_and_final_video(self):
        """Image uploads should hide audio and show the generated MP4."""

        with patch(
            "acestep.ui.gradio.events.wiring.simple_media_outputs.export_simple_video_artifacts",
            return_value=SimpleVideoArtifacts(
                video_path="song.mp4",
                image_path="video_image.png",
            ),
        ) as mock_export:
            updates = list(
                build_simple_media_preview(
                    "C:/temp/gradio/song.flac",
                    "done",
                    "cover.png",
                    "720p",
                    ["G:/ACE_Step_v1/ACE-Step_Premium/outputs/0011/song.flac"],
                )
            )

        self.assertEqual(len(updates), 2)
        mock_export.assert_called_once_with(
            "G:/ACE_Step_v1/ACE-Step_Premium/outputs/0011/song.flac",
            "cover.png",
            "720p",
        )
        self.assertIn("Creating MP4", updates[0][2])
        audio_update, video_update, status, files_update = updates[1]
        self.assertFalse(audio_update["visible"])
        self.assertEqual(video_update["value"], "song.mp4")
        self.assertTrue(video_update["visible"])
        self.assertIn("MP4 video ready", status)
        self.assertIn("Outputs are saved", status)
        self.assertIn("video_image.png", status)
        self.assertTrue(files_update["visible"])
        self.assertIn("song.mp4", files_update["value"])
        self.assertIn("video_image.png", files_update["value"])

    def test_preview_exports_video_for_each_generated_song(self):
        """A supplied image should create one MP4 for every generated song."""

        def fake_export(audio_path: str, image_path: str, resolution: str):
            stem = audio_path.rsplit("/", 1)[-1].removesuffix(".flac")
            return SimpleVideoArtifacts(
                video_path=f"G:/outputs/0011/{stem}_{resolution}.mp4",
                image_path="G:/outputs/0011/video_image.png",
            )

        with patch(
            "acestep.ui.gradio.events.wiring.simple_media_outputs.export_simple_video_artifacts",
            side_effect=fake_export,
        ) as mock_export:
            updates = list(
                build_simple_media_preview(
                    "C:/temp/gradio/song-1.flac",
                    "done",
                    "cover.png",
                    "1080p",
                    [
                        "G:/outputs/0011/song-1.flac",
                        "G:/outputs/0011/song-1.mp3",
                        "G:/outputs/0011/song-2.flac",
                        "G:/outputs/0011/song-2.mp3",
                    ],
                )
            )

        self.assertEqual(mock_export.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in mock_export.call_args_list],
            ["G:/outputs/0011/song-1.flac", "G:/outputs/0011/song-2.flac"],
        )
        self.assertIn("Creating 2 MP4 videos", updates[0][2])
        audio_update, video_update, status, files_update = updates[-1]
        self.assertFalse(audio_update["visible"])
        self.assertEqual(video_update["value"], "G:/outputs/0011/song-1_1080p.mp4")
        self.assertIn("2 MP4 videos ready", status)
        self.assertIn("Preview MP4", status)
        self.assertIn("G:/outputs/0011/song-1_1080p.mp4", files_update["value"])
        self.assertIn("G:/outputs/0011/song-2_1080p.mp4", files_update["value"])

    def test_generated_files_update_shows_all_paths(self):
        """Simple tab should expose all generated files for multi-song runs."""

        update = build_simple_generated_files_update(
            [
                {"path": "G:/outputs/song-1.flac"},
                {"path": "G:/outputs/song-2.flac"},
            ]
        )

        self.assertTrue(update["visible"])
        self.assertEqual(
            update["value"],
            ["G:/outputs/song-1.flac", "G:/outputs/song-2.flac"],
        )


if __name__ == "__main__":
    unittest.main()
