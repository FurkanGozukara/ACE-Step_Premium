"""Tests for dataset audio duration helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.training.dataset_builder_modules.audio_io import get_audio_duration
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class AudioIoTests(unittest.TestCase):
    """Tests for duration probing order."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_get_audio_duration_uses_soundfile_first(self) -> None:
        """Supported formats should not invoke the ffprobe fallback."""

        set_safe_roots(["C:\\"])
        with patch(
            "acestep.training.dataset_builder_modules.audio_io._duration_from_soundfile",
            return_value=12,
        ) as soundfile_probe, patch(
            "acestep.training.dataset_builder_modules.audio_io._duration_from_ffprobe"
        ) as ffprobe_probe:
            duration = get_audio_duration("C:\\temp\\sample.wav")

        self.assertEqual(12, duration)
        soundfile_probe.assert_called_once()
        ffprobe_probe.assert_not_called()

    def test_get_audio_duration_falls_back_to_ffprobe(self) -> None:
        """MP3-style paths should use ffprobe when soundfile cannot read them."""

        set_safe_roots(["C:\\"])
        with patch(
            "acestep.training.dataset_builder_modules.audio_io._duration_from_soundfile",
            return_value=None,
        ), patch(
            "acestep.training.dataset_builder_modules.audio_io._duration_from_ffprobe",
            return_value=34,
        ) as ffprobe_probe:
            duration = get_audio_duration("C:\\temp\\sample.mp3")

        self.assertEqual(34, duration)
        ffprobe_probe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
