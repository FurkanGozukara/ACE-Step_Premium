"""Tests for Remix source range resolution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.results.remix_source_range import (
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

    def test_valid_remix_range_returns_trimmed_path(self) -> None:
        """A valid Remix range should be trimmed and returned for generation."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ), patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_trim_source_range_preview",
                return_value="trimmed_source.wav",
            ) as trim_mock:
                result = resolve_remix_source_range_audio(
                    "cover",
                    str(source),
                    5.0,
                    12.0,
                )

        self.assertEqual(result, "trimmed_source.wav")
        trim_mock.assert_called_once_with(str(source), 5.0, 7.0)

    def test_full_source_range_keeps_original_source(self) -> None:
        """A default 0 to -1 range should not create a needless trim."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ), patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_trim_source_range_preview",
            ) as trim_mock:
                result = resolve_remix_source_range_audio(
                    "cover",
                    str(source),
                    0.0,
                    -1,
                )

        self.assertEqual(result, str(source))
        trim_mock.assert_not_called()

    def test_invalid_range_keeps_original_source(self) -> None:
        """Invalid in-progress start/end values should not raise or trim."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"fake")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=30.0,
            ), patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_trim_source_range_preview",
            ) as trim_mock:
                result = resolve_remix_source_range_audio(
                    "cover",
                    str(source),
                    12.0,
                    5.0,
                )

        self.assertEqual(result, str(source))
        trim_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
