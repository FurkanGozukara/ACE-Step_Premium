"""Unit tests for Gradio media upload preview helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from acestep.ui.gradio.events.wiring.audio_processing_source_paths import (
    effective_single_file_input,
)
from acestep.ui.gradio.events.wiring.audio_processing_wiring import (
    _preview_upload as preview_audio_processing_upload,
)
from acestep.ui.gradio.events.wiring.generation_upload_handlers import (
    finalize_src_audio_upload,
    handle_src_audio_upload,
    use_generated_result_as_source,
)
from acestep.ui.gradio.events.wiring.media_upload_preview import (
    PREVIEW_AUDIO_SECONDS,
    PREVIEW_AUDIO_TIMEOUT_SECONDS,
    extract_audio_preview,
    preview_audio_purpose_upload,
    preview_audio_purpose_upload_direct,
    preview_video_upload,
)
from acestep.ui.gradio.events.wiring.sam_audio_action_helpers import (
    preview_upload as preview_sam_upload,
)
from acestep.ui.gradio.events.wiring.sam_audio_processing import (
    _effective_single_file_input as _sam_effective_single_file_input,
)
from acestep.ui.gradio.media_upload_values import (
    latest_upload_path,
    resolve_effective_source_audio,
)


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

    def test_effective_source_uses_original_upload_when_preview_is_unedited(self):
        """Initial video preview extraction should not replace the original source."""

        self.assertEqual(
            resolve_effective_source_audio(
                "song_video.mp4",
                "song_video_audio_preview.wav",
                "song_video_audio_preview.wav",
            ),
            "song_video.mp4",
        )

    def test_effective_source_uses_trimmed_preview_when_preview_changes(self):
        """Edited Source Audio Preview should become the generation source."""

        self.assertEqual(
            resolve_effective_source_audio(
                "song_video.mp4",
                "trimmed_preview.wav",
                "song_video_audio_preview.wav",
            ),
            "trimmed_preview.wav",
        )

    def test_audio_processing_prefers_trimmed_upload_preview(self):
        """Edited Audio Processing upload preview should become the processing source."""

        self.assertEqual(
            effective_single_file_input("original.wav", "trimmed_preview.wav"),
            "trimmed_preview.wav",
        )

    def test_audio_processing_falls_back_to_upload_without_audio_preview(self):
        """Video uploads without an audio preview should still process the original upload."""

        self.assertEqual(
            effective_single_file_input("source_video.mp4", None),
            "source_video.mp4",
        )

    def test_sam_prefers_trimmed_upload_preview(self):
        """Edited SAM upload preview should become the processing source."""

        self.assertEqual(
            _sam_effective_single_file_input("original.wav", "trimmed_preview.wav"),
            "trimmed_preview.wav",
        )

    def test_sam_falls_back_to_upload_without_audio_preview(self):
        """Video-only SAM uploads should still process the original upload."""

        self.assertEqual(
            _sam_effective_single_file_input("source_video.mp4", None),
            "source_video.mp4",
        )

    def test_audio_purpose_upload_shows_audio_for_audio_file(self):
        """Audio files should show only the audio preview."""

        audio_update, video_update = preview_audio_purpose_upload("song.wav")

        self.assertEqual(audio_update.get("value"), "song.wav")
        self.assertTrue(audio_update.get("visible"))
        self.assertIsNone(video_update.get("value"))
        self.assertFalse(video_update.get("visible"))

    @patch("acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview")
    def test_direct_audio_purpose_upload_shows_video_without_extracting_audio(
        self,
        extract_mock,
    ):
        """Fast upload previews should show video immediately without ffmpeg."""

        audio_update, video_update = preview_audio_purpose_upload_direct("clip.mp4")

        self.assertIsNone(audio_update.get("value"))
        self.assertFalse(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "clip.mp4")
        self.assertTrue(video_update.get("visible"))
        extract_mock.assert_not_called()

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
        extract_mock.assert_called_once_with("clip.mp4", progress=ANY)

    @patch(
        "acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview",
        return_value="new_audio_preview.wav",
    )
    def test_audio_purpose_upload_uses_newest_stale_single_file_value(self, extract_mock):
        """Audio-purpose previews should tolerate Gradio stale file lists."""

        audio_update, video_update = preview_audio_purpose_upload(["old.wav", "new.mp4"])

        self.assertEqual(audio_update.get("value"), "new_audio_preview.wav")
        self.assertEqual(video_update.get("value"), "new.mp4")
        extract_mock.assert_called_once_with("new.mp4", progress=ANY)

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
        extract_mock.assert_called_once_with("clip.mp4", progress=ANY)
        warning_mock.assert_called_once()

    @patch(
        "acestep.ui.gradio.events.wiring.generation_upload_handlers.gen_h."
        "handle_extract_src_audio_change",
    )
    @patch("acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview")
    def test_source_upload_fast_stage_skips_extraction_and_duration(
        self,
        extract_mock,
        duration_mock,
    ):
        """Initial Source Audio upload updates should not block on slow work."""

        audio_update, video_update, duration_update, original = handle_src_audio_upload(
            "clip.mp4",
            "Complete",
        )

        self.assertIsNone(audio_update.get("value"))
        self.assertFalse(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "clip.mp4")
        self.assertTrue(video_update.get("visible"))
        self.assertNotIn("value", duration_update)
        self.assertIsNone(original)
        extract_mock.assert_not_called()
        duration_mock.assert_not_called()

    @patch(
        "acestep.ui.gradio.events.wiring.generation_upload_handlers.gen_h."
        "handle_extract_src_audio_change",
        return_value={"value": 42.0, "__type__": "update"},
    )
    @patch(
        "acestep.ui.gradio.events.wiring.media_upload_preview.extract_audio_preview",
        return_value="clip_audio_preview.wav",
    )
    def test_source_upload_finalize_extracts_preview_with_progress(
        self,
        extract_mock,
        duration_mock,
    ):
        """Source upload finalize should extract audio and report progress."""

        progress = MagicMock()
        audio_update, video_update, duration_update, original = finalize_src_audio_upload(
            "clip.mp4",
            "Complete",
            progress=progress,
        )

        self.assertEqual(audio_update.get("value"), "clip_audio_preview.wav")
        self.assertTrue(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "clip.mp4")
        self.assertTrue(video_update.get("visible"))
        self.assertEqual(duration_update.get("value"), 42.0)
        self.assertEqual(original, "clip_audio_preview.wav")
        extract_mock.assert_called_once_with("clip.mp4", progress=progress)
        duration_mock.assert_called_once_with("clip.mp4", "Complete")
        self.assertGreaterEqual(progress.call_count, 3)

    @patch("acestep.ui.gradio.events.wiring.generation_upload_handlers.gr.Info")
    @patch(
        "acestep.ui.gradio.events.wiring.generation_upload_handlers."
        "preview_source_range",
        return_value=(
            {"value": "range.wav", "visible": True, "__type__": "update"},
            {"value": None, "visible": False, "__type__": "update"},
        ),
    )
    @patch(
        "acestep.ui.gradio.events.wiring.generation_upload_handlers.gen_h."
        "handle_extract_src_audio_change",
        return_value={"value": 42.0, "__type__": "update"},
    )
    def test_use_generated_result_as_source_updates_source_preview_and_remix_range(
        self,
        duration_mock,
        range_preview_mock,
        info_mock,
    ):
        """Inline generated audio should replace the Remix source and full range."""

        (
            src_audio,
            audio_update,
            video_update,
            duration_update,
            original,
            start_update,
            end_update,
            range_audio_update,
            range_video_update,
        ) = (
            use_generated_result_as_source("generated.wav", "Remix", 12.0, 20.0)
        )

        self.assertEqual(src_audio, ["generated.wav"])
        self.assertEqual(audio_update.get("value"), "generated.wav")
        self.assertTrue(audio_update.get("visible"))
        self.assertIsNone(video_update.get("value"))
        self.assertFalse(video_update.get("visible"))
        self.assertEqual(duration_update.get("value"), 42.0)
        self.assertIsNone(original)
        self.assertEqual(start_update.get("value"), 0.0)
        self.assertEqual(end_update.get("value"), -1)
        self.assertEqual(range_audio_update.get("value"), "range.wav")
        self.assertTrue(range_audio_update.get("visible"))
        self.assertIsNone(range_video_update.get("value"))
        self.assertFalse(range_video_update.get("visible"))
        duration_mock.assert_called_once_with("generated.wav", "Remix")
        range_preview_mock.assert_called_once_with(
            ["generated.wav"],
            "generated.wav",
            None,
            0.0,
            -1,
            "Remix",
        )
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.wiring.generation_upload_handlers.gr.Info")
    @patch(
        "acestep.ui.gradio.events.wiring.generation_upload_handlers."
        "preview_source_range",
        return_value=(
            {"value": "complete_range.wav", "visible": True, "__type__": "update"},
            {"value": None, "visible": False, "__type__": "update"},
        ),
    )
    @patch(
        "acestep.ui.gradio.events.wiring.generation_upload_handlers.gen_h."
        "handle_extract_src_audio_change",
        return_value={"value": 42.0, "__type__": "update"},
    )
    def test_use_generated_result_as_source_preserves_non_remix_range(
        self,
        duration_mock,
        range_preview_mock,
        info_mock,
    ):
        """Repaint/Lego/Complete ranges should be preserved and previewed."""

        updates = use_generated_result_as_source("generated.wav", "Complete", 12.0, 20.0)

        self.assertEqual(updates[5], {"__type__": "update"})
        self.assertEqual(updates[6], {"__type__": "update"})
        self.assertEqual(updates[7].get("value"), "complete_range.wav")
        self.assertTrue(updates[7].get("visible"))
        duration_mock.assert_called_once_with("generated.wav", "Complete")
        range_preview_mock.assert_called_once_with(
            ["generated.wav"],
            "generated.wav",
            None,
            12.0,
            20.0,
            "Complete",
        )
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.wiring.generation_upload_handlers.gr.Warning")
    def test_use_generated_result_as_source_without_audio_skips_updates(
        self,
        warning_mock,
    ):
        """Clicking the source-copy action before generation should be a no-op."""

        updates = use_generated_result_as_source(None, "Remix")

        self.assertEqual(len(updates), 9)
        for update in updates:
            self.assertEqual(update, {"__type__": "update"})
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

    def test_audio_processing_preview_can_be_disabled_for_large_video_uploads(self):
        """Disabled Audio Processing preview should avoid Gradio media outputs."""

        audio_update, video_update, status = preview_audio_processing_upload(
            "huge_concert.mkv",
            True,
        )

        self.assertIsNone(audio_update.get("value"))
        self.assertFalse(audio_update.get("visible"))
        self.assertIsNone(video_update.get("value"))
        self.assertFalse(video_update.get("visible"))
        self.assertIn("Upload preview disabled", status)
        self.assertIn("huge_concert.mkv", status)

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

        progress = MagicMock()

        result = extract_audio_preview("C:/media/long_clip.mp4", progress=progress)

        self.assertEqual(result, "C:/tmp/acestep_preview/long_clip_audio_preview.wav")
        command = run_mock.call_args.args[0]
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], str(PREVIEW_AUDIO_SECONDS))
        self.assertEqual(run_mock.call_args.kwargs["timeout"], PREVIEW_AUDIO_TIMEOUT_SECONDS)
        self.assertGreaterEqual(progress.call_count, 2)


if __name__ == "__main__":
    unittest.main()
