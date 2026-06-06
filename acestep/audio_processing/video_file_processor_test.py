"""Tests for video routing in single-file audio processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from acestep.audio_processing.file_processor import process_media_file
from acestep.audio_processing.pipeline import ProcessedAudio
from acestep.audio_processing.settings import AudioProcessingSettings


class AudioProcessingVideoFileProcessorTests(unittest.TestCase):
    """Verify video processing chooses copy or Auto-Editor paths correctly."""

    def test_video_without_trim_replaces_audio_without_auto_editor(self) -> None:
        """Audio-only video changes should copy video and replace audio."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            output = root / "out"
            audio_path = str(output / "source_processed.wav")
            video_path = str(output / "source_processed.mp4")

            with (
                patch(
                    "acestep.audio_processing.file_processor.read_media_audio",
                    return_value=(_test_tone(1000, seconds=0.1), 1000),
                ),
                patch(
                    "acestep.audio_processing.file_processor.process_audio_array",
                    return_value=_processed_audio(),
                ) as process_audio,
                patch(
                    "acestep.audio_processing.file_processor.save_processed_audio",
                    return_value=audio_path,
                ),
                patch(
                    "acestep.audio_processing.file_processor.mux_video_with_audio",
                    return_value=video_path,
                ) as mux_video,
                patch(
                    "acestep.audio_processing.file_processor.process_video_with_auto_editor",
                ) as auto_editor_video,
            ):
                result = process_media_file(source, output, AudioProcessingSettings())

            self.assertEqual(video_path, result.video_path)
            process_audio.assert_called_once()
            mux_video.assert_called_once()
            auto_editor_video.assert_not_called()

    def test_video_trim_uses_auto_editor_video_processing(self) -> None:
        """Trim-enabled video should be rendered by Auto-Editor as a video."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            output = root / "out"
            audio_path = str(output / "source_processed.wav")
            video_path = str(output / "source_processed.mp4")
            video_result = (_processed_audio(trimmed=True), audio_path, video_path)

            with (
                patch(
                    "acestep.audio_processing.file_processor.read_media_audio",
                    return_value=(_test_tone(1000, seconds=0.1), 1000),
                ),
                patch(
                    "acestep.audio_processing.file_processor.process_video_with_auto_editor",
                    return_value=video_result,
                ) as auto_editor_video,
                patch(
                    "acestep.audio_processing.file_processor.process_audio_array",
                ) as process_audio,
                patch(
                    "acestep.audio_processing.file_processor.mux_video_with_audio",
                ) as mux_video,
            ):
                result = process_media_file(
                    source,
                    output,
                    AudioProcessingSettings(trim_empty_output=True),
                )

            self.assertEqual(video_path, result.video_path)
            auto_editor_video.assert_called_once()
            process_audio.assert_not_called()
            mux_video.assert_not_called()

    def test_video_export_audio_only_skips_video_output(self) -> None:
        """Export-only-audio should not mux or render video for video inputs."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            output = root / "out"
            audio_path = str(output / "source_processed.wav")

            with (
                patch(
                    "acestep.audio_processing.file_processor.read_media_audio",
                    return_value=(_test_tone(1000, seconds=0.1), 1000),
                ),
                patch(
                    "acestep.audio_processing.file_processor.process_audio_array",
                    return_value=_processed_audio(),
                ) as process_audio,
                patch(
                    "acestep.audio_processing.file_processor.save_processed_audio",
                    return_value=audio_path,
                ),
                patch(
                    "acestep.audio_processing.file_processor.mux_video_with_audio",
                ) as mux_video,
                patch(
                    "acestep.audio_processing.file_processor.process_video_with_auto_editor",
                ) as auto_editor_video,
            ):
                result = process_media_file(
                    source,
                    output,
                    AudioProcessingSettings(
                        export_audio_only=True,
                        trim_empty_output=True,
                    ),
                )

            self.assertIsNone(result.video_path)
            self.assertEqual(audio_path, result.audio_path)
            process_audio.assert_called_once()
            process_args, _process_kwargs = process_audio.call_args
            self.assertTrue(process_args[2].trim_empty_output)
            mux_video.assert_not_called()
            auto_editor_video.assert_not_called()


def _processed_audio(trimmed: bool = False) -> ProcessedAudio:
    """Return a deterministic in-memory processed-audio result."""

    audio = _test_tone(1000, seconds=0.1)
    return ProcessedAudio(
        before=audio,
        after=audio,
        sample_rate=1000,
        lufs_before=-20.0,
        lufs_after=-18.0,
        duration_seconds=0.1,
        trim_metadata={
            "enabled": trimmed,
            "applied": trimmed,
            "reason": "auto_editor_video_trimmed" if trimmed else "disabled",
        },
    )


def _test_tone(sample_rate: int, seconds: float) -> np.ndarray:
    """Return deterministic stereo audio for mocked processing tests."""

    time = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = np.stack(
        [
            0.2 * np.sin(2 * np.pi * 220 * time),
            0.18 * np.sin(2 * np.pi * 330 * time),
        ],
        axis=1,
    )
    return audio.astype(np.float32)


if __name__ == "__main__":
    unittest.main()
