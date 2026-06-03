"""Unit tests for Gradio media upload preview helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.audio_processing_wiring import (
    _preview_upload as preview_audio_processing_upload,
)
from acestep.ui.gradio.events.wiring.media_upload_preview import (
    PREVIEW_AUDIO_SECONDS,
    PREVIEW_AUDIO_TIMEOUT_SECONDS,
    extract_audio_preview,
    preview_audio_purpose_upload,
    preview_video_upload,
)
from acestep.ui.gradio.events.wiring.sam_audio_action_helpers import (
    preview_upload as preview_sam_upload,
)
from acestep.ui.gradio.media_upload_values import latest_upload_path


class MediaUploadPreviewTests(unittest.TestCase):
    """Tests for audio-purpose and video-purpose upload previews."""

    def test_latest_upload_path_uses_newest_stale_single_file_value(self):
        """Gradio stale single-file lists should resolve to the newest upload."""

        self.assertEqual(latest_upload_path(["old.wav", "new.mp4"]), "new.mp4")
        self.assertEqual(
            latest_upload_path([{"path": "old.wav"}, {"path": "new.mp4"}]),
            "new.mp4",
        )
        self.assertEqual(
            latest_upload_path([SimpleNamespace(path="old.wav"), SimpleNamespace(name="new.mp4")]),
            "new.mp4",
        )

    def test_audio_purpose_upload_shows_audio_for_audio_file(self):
        """Audio files should show only the audio preview."""

        audio_update, video_update = preview_audio_purpose_upload("song.wav")

        self.assertEqual(audio_update.get("value"), "song.wav")
        self.assertTrue(audio_update.get("visible"))
        self.assertIsNone(video_update.get("value"))
        self.assertFalse(video_update.get("visible"))

    @patch(
        "acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview",
        return_value="clip_audio_preview.wav",
    )
    def test_audio_purpose_upload_extracts_audio_preview_for_video_file(self, extract_mock):
        """Video files in audio-purpose fields should show extracted audio and video."""

        audio_update, video_update = preview_audio_purpose_upload("clip.mp4")

        self.assertEqual(audio_update.get("value"), "clip_audio_preview.wav")
        self.assertTrue(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "clip.mp4")
        self.assertTrue(video_update.get("visible"))
        extract_mock.assert_called_once_with("clip.mp4")

    @patch(
        "acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview",
        return_value="new_audio_preview.wav",
    )
    def test_audio_purpose_upload_uses_newest_stale_single_file_value(self, extract_mock):
        """Audio-purpose previews should tolerate Gradio stale file lists."""

        audio_update, video_update = preview_audio_purpose_upload(["old.wav", "new.mp4"])

        self.assertEqual(audio_update.get("value"), "new_audio_preview.wav")
        self.assertEqual(video_update.get("value"), "new.mp4")
        extract_mock.assert_called_once_with("new.mp4")

    @patch("acestep.ui.gradio.events.wiring.media_upload_preview.gr.Warning")
    @patch(
        "acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview",
        side_effect=RuntimeError("decode failed"),
    )
    def test_audio_purpose_upload_keeps_video_preview_when_audio_extract_fails(
        self,
        extract_mock,
        warning_mock,
    ):
        """Video preview should remain visible if audio preview extraction fails."""

        audio_update, video_update = preview_audio_purpose_upload("clip.mp4")

        self.assertIsNone(audio_update.get("value"))
        self.assertFalse(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "clip.mp4")
        self.assertTrue(video_update.get("visible"))
        extract_mock.assert_called_once_with("clip.mp4")
        warning_mock.assert_called_once()

    def test_video_upload_shows_video_preview(self):
        """Video-only fields should show uploaded video previews."""

        update = preview_video_upload("mask.webm")

        self.assertEqual(update.get("value"), "mask.webm")
        self.assertTrue(update.get("visible"))

    def test_video_upload_uses_newest_stale_single_file_value(self):
        """Video-only previews should tolerate Gradio stale file lists."""

        update = preview_video_upload(["old.mp4", "mask.webm"])

        self.assertEqual(update.get("value"), "mask.webm")
        self.assertTrue(update.get("visible"))

    def test_audio_processing_preview_uses_newest_stale_single_file_value(self):
        """Audio Processing upload preview should tolerate stale Gradio lists."""

        audio_update, video_update, status = preview_audio_processing_upload(
            ["old.wav", "new.mp4"]
        )

        self.assertFalse(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "new.mp4")
        self.assertIn("new.mp4", status)

    def test_sam_preview_uses_newest_stale_single_file_value(self):
        """SAM upload preview should tolerate stale Gradio lists."""

        audio_update, video_update, status = preview_sam_upload(["old.wav", "new.mp4"])

        self.assertFalse(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "new.mp4")
        self.assertIn("new.mp4", status)

    @patch("acestep.ui.gradio.events.wiring.media_upload_preview.tempfile.mkdtemp")
    @patch("acestep.ui.gradio.events.wiring.media_upload_preview.subprocess.run")
    def test_extract_audio_preview_limits_ffmpeg_decode_duration(
        self,
        run_mock,
        mkdtemp_mock,
    ):
        """Preview extraction should decode a bounded slice, not the whole video."""

        mkdtemp_mock.return_value = "C:/tmp/acestep_preview"

        result = extract_audio_preview("C:/media/long_clip.mp4")

        self.assertEqual(result, "C:/tmp/acestep_preview/long_clip_audio_preview.wav")
        command = run_mock.call_args.args[0]
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], str(PREVIEW_AUDIO_SECONDS))
        self.assertEqual(run_mock.call_args.kwargs["timeout"], PREVIEW_AUDIO_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
