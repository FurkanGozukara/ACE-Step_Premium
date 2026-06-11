"""Tests for shared silence trimming helpers."""

import unittest
from unittest.mock import patch

import torch

from acestep.audio_processing.silence_trim import (
    clamp_trim_threshold_db,
    trim_silent_edges,
)


class TestSilenceTrim(unittest.TestCase):
    """Verify auto-editor based trim behavior."""

    def test_cuts_inactive_sections_from_auto_editor_spans(self) -> None:
        """Detected auto-editor spans are concatenated in output order."""

        audio = torch.tensor([[0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0]])

        with patch(
            "acestep.audio_processing.auto_editor_trim._detect_spans_with_auto_editor",
            return_value=[(1, 3), (4, 6)],
        ):
            result = trim_silent_edges(audio, sample_rate=10, enabled=True)

        self.assertTrue(result.metadata["applied"])
        self.assertEqual("auto_editor_trimmed", result.metadata["reason"])
        self.assertEqual(2, result.metadata["segments_count"])
        self.assertEqual((1, 4), tuple(result.audio.shape))
        self.assertTrue(torch.equal(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), result.audio))

    def test_no_active_segments_keeps_original_audio(self) -> None:
        """No detected segments are kept to avoid writing empty audio files."""

        audio = torch.zeros(1, 8)

        with patch(
            "acestep.audio_processing.auto_editor_trim._detect_spans_with_auto_editor",
            return_value=[],
        ):
            result = trim_silent_edges(audio, sample_rate=8, enabled=True)

        self.assertFalse(result.metadata["applied"])
        self.assertEqual("no_active_segments", result.metadata["reason"])
        self.assertIs(result.audio, audio)

    def test_disabled_trim_reports_metadata_without_slicing(self) -> None:
        """Disabled trim returns the original tensor and clear metadata."""

        audio = torch.ones(1, 4)

        result = trim_silent_edges(
            audio,
            sample_rate=4,
            enabled=False,
        )

        self.assertFalse(result.metadata["applied"])
        self.assertEqual("disabled", result.metadata["reason"])
        self.assertEqual(1.0, result.metadata["trimmed_duration_seconds"])
        self.assertIs(result.audio, audio)

    def test_threshold_is_clamped_to_supported_range(self) -> None:
        """Threshold values are limited to the shared UI/backend range."""

        self.assertEqual(-100.0, clamp_trim_threshold_db(-120.0))
        self.assertEqual(0.0, clamp_trim_threshold_db(120.0))

    def test_threshold_override_reaches_auto_editor_settings(self) -> None:
        """Saved trim thresholds should be applied to auto-editor detection."""

        audio = torch.ones(1, 4)

        with patch(
            "acestep.audio_processing.auto_editor_trim._detect_spans_with_auto_editor",
            return_value=[(0, 4)],
        ) as detect_mock:
            result = trim_silent_edges(
                audio,
                sample_rate=4,
                enabled=True,
                threshold_db=-55.0,
            )

        settings = detect_mock.call_args.kwargs["settings"]
        self.assertEqual(-55.0, settings.threshold_db)
        self.assertEqual(-55.0, result.metadata["settings"]["threshold_db"])


if __name__ == "__main__":
    unittest.main()
