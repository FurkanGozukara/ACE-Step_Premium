"""Tests for SAM-Audio Gradio processing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from acestep.sam_audio_segment.settings import SAM_AUDIO_PRESET_KEYS, SamAudioSettings
from acestep.ui.gradio.events.wiring.sam_audio_processing import (
    _process_single_in_process,
    _settings_from_input_values,
)


class _Artifact:
    """Small artifact object matching the service return contract."""

    def __init__(self) -> None:
        self.target_audio_path = "target.wav"
        self.residual_audio_path = "residual.wav"
        self.target_video_path = None
        self.metadata_path = "metadata.json"

    def file_list(self) -> list[str]:
        """Return generated file paths."""

        return [self.target_audio_path, self.residual_audio_path, self.metadata_path]


class SamAudioProcessingTests(unittest.TestCase):
    """Verify same-process SAM-Audio processing uses the model cache."""

    def test_single_in_process_uses_cached_service_without_unloading(self) -> None:
        """Manual same-process processing should not unload the model per request."""

        service = MagicMock()
        service.process_file.return_value = _Artifact()
        settings = SamAudioSettings(subprocess=False)
        progress_callback = object()
        cache_calls = []

        @contextmanager
        def _cached_service(*args, **kwargs):
            """Yield the fake cached service and record cache inputs."""

            cache_calls.append((args, kwargs))
            yield service

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "acestep.ui.gradio.events.wiring.sam_audio_processing.cached_sam_audio_service",
            new=_cached_service,
        ):
            artifacts, files = _process_single_in_process(
                "C:/music/My Song.wav",
                None,
                Path(temp_dir),
                settings,
                progress_callback,
            )

        self.assertEqual("target.wav", artifacts["target_audio_path"])
        self.assertEqual(["target.wav", "residual.wav", "metadata.json"], files)
        self.assertEqual(1, len(cache_calls))
        self.assertEqual(settings, cache_calls[0][0][0])
        self.assertIs(progress_callback, cache_calls[0][1]["progress_callback"])
        service.unload.assert_not_called()
        _, _, kwargs = service.process_file.mock_calls[0]
        self.assertEqual("My Song_sam", kwargs["output_stem"])
        self.assertIsNone(kwargs["mask_video_path"])

    def test_settings_from_input_values_uses_audio_processing_trim_tail(self) -> None:
        """Standalone SAM handlers should use shared Audio Processing trim values."""

        sam_values = {key: None for key in SAM_AUDIO_PRESET_KEYS}
        sam_values.update(
            {
                "sam_trim_empty_output": True,
                "sam_trim_threshold_db": -60.0,
                "sam_output_format": "mp3",
            }
        )
        settings = _settings_from_input_values(
            (
                *[sam_values[key] for key in SAM_AUDIO_PRESET_KEYS],
                -35.0,
                0.8,
                12,
                6,
            )
        )

        self.assertTrue(settings.trim_empty_output)
        self.assertEqual(-35.0, settings.trim_threshold_db)
        self.assertEqual(0.8, settings.trim_margin_seconds)
        self.assertEqual(12, settings.trim_mincut)
        self.assertEqual(6, settings.trim_minclip)


if __name__ == "__main__":
    unittest.main()
