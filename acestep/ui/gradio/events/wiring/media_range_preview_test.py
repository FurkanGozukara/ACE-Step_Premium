"""Tests for source range preview helpers."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import DEFAULT, patch

from acestep.ui.gradio.events.wiring.media_range_preview import (
    SOURCE_RANGE_PREVIEW_MAX_SECONDS,
    preview_source_range,
)
from acestep.ui.gradio.events.wiring.media_range_preview_trim import (
    SOURCE_RANGE_PREVIEW_TIMEOUT_SECONDS,
    trim_source_range_preview,
)


class MediaRangePreviewTests(unittest.TestCase):
    """Verify robust source range preview behavior."""

    def test_audio_range_preview_trims_valid_range(self) -> None:
        """Valid audio ranges should return a visible audio preview."""

        with tempfile.TemporaryDirectory() as tmpdir:
            source = _touch(Path(tmpdir) / "song.wav")
            with _preview_patches(duration=120.0, preview_path="range.wav") as trim_mock:
                audio_update, video_update = preview_source_range(
                    str(source),
                    None,
                    None,
                    5.0,
                    15.0,
                    "Repaint",
                )

        self.assertEqual(audio_update.get("value"), "range.wav")
        self.assertTrue(audio_update.get("visible"))
        self.assertIsNone(video_update.get("value"))
        self.assertFalse(video_update.get("visible"))
        trim_mock.assert_called_once_with(str(source), 5.0, 10.0)

    def test_video_range_preview_returns_video_update(self) -> None:
        """Valid video ranges should return a visible video preview."""

        with tempfile.TemporaryDirectory() as tmpdir:
            source = _touch(Path(tmpdir) / "clip.mp4")
            with _preview_patches(duration=30.0, preview_path="range.mp4") as trim_mock:
                audio_update, video_update = preview_source_range(
                    str(source),
                    None,
                    None,
                    2.0,
                    7.0,
                    "Lego",
                )

        self.assertIsNone(audio_update.get("value"))
        self.assertFalse(audio_update.get("visible"))
        self.assertEqual(video_update.get("value"), "range.mp4")
        self.assertTrue(video_update.get("visible"))
        trim_mock.assert_called_once_with(str(source), 2.0, 5.0)

    def test_negative_end_uses_source_end_with_preview_cap(self) -> None:
        """End ``-1`` should preview from start through source end, capped."""

        with tempfile.TemporaryDirectory() as tmpdir:
            source = _touch(Path(tmpdir) / "song.wav")
            with _preview_patches(duration=200.0, preview_path="range.wav") as trim_mock:
                audio_update, _video_update = preview_source_range(
                    str(source),
                    None,
                    None,
                    20.0,
                    -1,
                    "Complete",
                )

        self.assertTrue(audio_update.get("visible"))
        trim_mock.assert_called_once_with(
            str(source),
            20.0,
            SOURCE_RANGE_PREVIEW_MAX_SECONDS,
        )

    def test_invalid_values_hide_preview_without_trimming(self) -> None:
        """Invalid or in-progress numeric values should not raise or trim."""

        with tempfile.TemporaryDirectory() as tmpdir:
            source = _touch(Path(tmpdir) / "song.wav")
            with _preview_patches(duration=120.0, preview_path="range.wav") as trim_mock:
                cases = [
                    (None, 10.0),
                    (10.0, None),
                    (-1.0, 10.0),
                    (10.0, 10.0),
                    (15.0, 10.0),
                    (130.0, -1),
                ]
                for start, end in cases:
                    with self.subTest(start=start, end=end):
                        audio_update, video_update = preview_source_range(
                            str(source),
                            None,
                            None,
                            start,
                            end,
                            "Repaint",
                        )
                        self.assertFalse(audio_update.get("visible"))
                        self.assertFalse(video_update.get("visible"))

        trim_mock.assert_not_called()

    def test_edited_source_audio_preview_is_used_for_range_preview(self) -> None:
        """Edited Source Audio Preview should become the preview source."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = _touch(root / "original.wav")
            original_preview = _touch(root / "original_preview.wav")
            edited_preview = _touch(root / "trimmed_preview.wav")
            with _preview_patches(duration=20.0, preview_path="range.wav") as trim_mock:
                audio_update, _video_update = preview_source_range(
                    str(original),
                    str(edited_preview),
                    str(original_preview),
                    1.0,
                    3.0,
                    "Remix",
                )

        self.assertTrue(audio_update.get("visible"))
        trim_mock.assert_called_once_with(str(edited_preview), 1.0, 2.0)

    def test_unsupported_mode_hides_preview_without_trimming(self) -> None:
        """Modes without source range preview support should hide the preview."""

        with tempfile.TemporaryDirectory() as tmpdir:
            source = _touch(Path(tmpdir) / "song.wav")
            with _preview_patches(duration=20.0, preview_path="range.wav") as trim_mock:
                audio_update, video_update = preview_source_range(
                    str(source),
                    None,
                    None,
                    1.0,
                    3.0,
                    "Simple",
                )

        self.assertFalse(audio_update.get("visible"))
        self.assertFalse(video_update.get("visible"))
        trim_mock.assert_not_called()


class MediaRangePreviewTrimTests(unittest.TestCase):
    """Verify ffmpeg trim commands for range previews."""

    @patch("acestep.ui.gradio.events.wiring.media_range_preview_trim.tempfile.mkdtemp")
    @patch("acestep.ui.gradio.events.wiring.media_range_preview_trim.subprocess.run")
    def test_audio_trim_uses_selected_start_and_duration(
        self,
        run_mock,
        mkdtemp_mock,
    ) -> None:
        """Audio preview trim should pass selected start and duration to ffmpeg."""

        mkdtemp_mock.return_value = "C:/tmp/acestep_range"

        result = trim_source_range_preview("C:/media/song.wav", 4.5, 8.25)

        self.assertEqual(result, "C:/tmp/acestep_range/song_range_preview.wav")
        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "4.500")
        self.assertEqual(command[command.index("-t") + 1], "8.250")
        self.assertIn("-vn", command)
        self.assertEqual(
            run_mock.call_args.kwargs["timeout"],
            SOURCE_RANGE_PREVIEW_TIMEOUT_SECONDS,
        )

    @patch("acestep.ui.gradio.events.wiring.media_range_preview_trim.tempfile.mkdtemp")
    @patch("acestep.ui.gradio.events.wiring.media_range_preview_trim.subprocess.run")
    def test_video_trim_keeps_video_and_optional_audio_streams(
        self,
        run_mock,
        mkdtemp_mock,
    ) -> None:
        """Video preview trim should produce an MP4 with video and optional audio."""

        mkdtemp_mock.return_value = "C:/tmp/acestep_range"

        result = trim_source_range_preview("C:/media/clip.mp4", 1.0, 2.5)

        self.assertEqual(result, "C:/tmp/acestep_range/clip_range_preview.mp4")
        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "1.000")
        self.assertEqual(command[command.index("-t") + 1], "2.500")
        self.assertIn("0:v:0", command)
        self.assertIn("0:a:0?", command)
        self.assertIn("libx264", command)


def _touch(path: Path) -> Path:
    """Create a tiny media placeholder file and return its path."""

    path.write_bytes(b"placeholder")
    return path


@contextmanager
def _preview_patches(duration: float, preview_path: str):
    """Patch duration probing and ffmpeg trim execution for preview tests."""

    with patch.multiple(
        "acestep.ui.gradio.events.wiring.media_range_preview",
        media_audio_duration_seconds=DEFAULT,
        trim_source_range_preview=DEFAULT,
    ) as mocks:
        mocks["media_audio_duration_seconds"].return_value = duration
        mocks["trim_source_range_preview"].return_value = preview_path
        yield mocks["trim_source_range_preview"]


if __name__ == "__main__":
    unittest.main()
