"""Tests for single-file audio and video processing."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from acestep.audio_processing.auto_editor_trim import SilenceTrimResult
from acestep.audio_processing.file_processor import process_media_file
from acestep.audio_processing.settings import AudioProcessingSettings


class AudioProcessingFileProcessorTests(unittest.TestCase):
    """Verify media files are processed into persisted artifacts."""

    def test_process_wav_writes_audio_and_metadata(self) -> None:
        """A WAV source should produce processed audio and JSON metadata."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            output = Path(temp_dir) / "out"
            _write_test_wav(source)

            result = process_media_file(source, output, AudioProcessingSettings())

            self.assertTrue(Path(result.audio_path).is_file())
            self.assertTrue(Path(result.metadata_path).is_file())
            payload = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(str(source.resolve()).replace("\\", "/"), payload["_meta"]["source_path"])
            self.assertIn("lufs_after", payload["metrics"])
            self.assertFalse(payload["trim"]["applied"])

    def test_process_wav_can_trim_silent_edges(self) -> None:
        """Audio Processing trim should shorten existing audio files when enabled."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            output = Path(temp_dir) / "out"
            _write_padded_wav(source)

            trimmed = torch.full((2, 24000), 0.2, dtype=torch.float32)
            metadata = {"enabled": True, "applied": True, "reason": "auto_editor_trimmed"}
            with patch(
                "acestep.audio_processing.pipeline.trim_silent_edges",
                return_value=SilenceTrimResult(trimmed, metadata),
            ):
                result = process_media_file(
                    source,
                    output,
                    AudioProcessingSettings(trim_empty_output=True),
                )

            info = sf.info(result.audio_path)
            payload = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
            self.assertLess(info.duration, 1.0)
            self.assertTrue(payload["trim"]["applied"])
            self.assertEqual("auto_editor_trimmed", payload["trim"]["reason"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for video processing")
    def test_process_video_writes_muxed_video(self) -> None:
        """A video source should return audio plus a video with processed audio."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.mp4"
            output = Path(temp_dir) / "out"
            _write_test_video(source)

            result = process_media_file(source, output, AudioProcessingSettings())

            self.assertTrue(Path(result.audio_path).is_file())
            self.assertIsNotNone(result.video_path)
            self.assertTrue(Path(result.video_path or "").is_file())


def _write_test_wav(path: Path, sample_rate: int = 48000) -> None:
    """Write deterministic stereo test audio to a WAV file."""

    time = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    audio = np.stack(
        [
            0.2 * np.sin(2 * np.pi * 220 * time),
            0.18 * np.sin(2 * np.pi * 330 * time),
        ],
        axis=1,
    ).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _write_padded_wav(path: Path, sample_rate: int = 48000) -> None:
    """Write a test WAV with silent leading and trailing edges."""

    silence = np.zeros((sample_rate // 4, 2), dtype=np.float32)
    tone = np.full((sample_rate // 2, 2), 0.2, dtype=np.float32)
    sf.write(path, np.concatenate([silence, tone, silence], axis=0), sample_rate)


def _write_test_video(path: Path) -> None:
    """Create a one-second synthetic MP4 with audio using ffmpeg."""

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x90:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1:sample_rate=48000",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
