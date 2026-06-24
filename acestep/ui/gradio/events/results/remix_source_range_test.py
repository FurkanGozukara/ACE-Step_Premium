"""Tests for Remix source range resolution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.results.remix_source_range import (
    remix_source_segment_for_clips,
    resolve_bounded_remix_source_range,
    resolve_remix_source_range_audio,
)


class RemixSourceRangeTests(unittest.TestCase):
    """Verify Remix source segment trimming decisions."""

    def test_non_remix_task_keeps_original_source(self) -> None:
        """Only Remix cover tasks should use the source-range trim path."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            result = resolve_remix_source_range_audio(
                "repaint",
                str(source),
                2.0,
                5.0,
            )

        self.assertEqual(result, str(source))

    def test_valid_remix_range_keeps_full_source_for_generation(self) -> None:
        """A valid Remix range should not trim the source sent to generation."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            result = resolve_remix_source_range_audio(
                "cover",
                str(source),
                5.0,
                12.0,
            )

        self.assertEqual(result, str(source))

    def test_valid_remix_range_resolves_post_generation_segment(self) -> None:
        """A bounded Remix range should be available for splice/area previews."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ):
                result = resolve_bounded_remix_source_range(
                    "cover",
                    str(source),
                    5.0,
                    12.0,
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.start, 5.0)
        self.assertEqual(result.duration, 7.0)

    def test_full_source_range_keeps_original_source(self) -> None:
        """A default 0 to -1 range should not create a needless trim."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ):
                result = resolve_remix_source_range_audio(
                    "cover",
                    str(source),
                    0.0,
                    -1,
                )

        self.assertEqual(result, str(source))
        self.assertIsNone(
            resolve_bounded_remix_source_range("cover", str(source), 0.0, -1)
        )

    def test_invalid_range_keeps_original_source(self) -> None:
        """Invalid in-progress start/end values should not raise or trim."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ):
                result = resolve_remix_source_range_audio(
                    "cover",
                    str(source),
                    12.0,
                    5.0,
                )

        self.assertEqual(result, str(source))
        self.assertIsNone(
            resolve_bounded_remix_source_range("cover", str(source), 12.0, 5.0)
        )

    def test_remix_clip_segment_uses_bounded_range_or_whole_source(self) -> None:
        """Latest Remixed Area should crop valid ranges and use whole audio otherwise."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ):
                bounded = remix_source_segment_for_clips("cover", str(source), 5.0, 12.0)
                full = remix_source_segment_for_clips("cover", str(source), 0.0, -1)

        self.assertEqual(bounded, (5.0, 7.0))
        self.assertEqual(full, (0.0, None))


if __name__ == "__main__":
    unittest.main()
