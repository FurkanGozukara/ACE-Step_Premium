"""Tests for dataset preprocessing audio loading helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from acestep.training.dataset_builder_modules import preprocess_audio
from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo


class PreprocessAudioTests(unittest.TestCase):
    """Tests for TorchCodec-free preprocessing audio loading."""

    def test_load_audio_stereo_uses_soundfile_first(self) -> None:
        """Supported files should not invoke the ffmpeg fallback."""

        mono_audio = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        with patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_soundfile",
            return_value=(mono_audio, 4),
        ) as soundfile_load, patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_ffmpeg"
        ) as ffmpeg_load:
            audio, sample_rate = load_audio_stereo("sample.wav", 4, 0.5)

        self.assertEqual(4, sample_rate)
        self.assertEqual((2, 2), tuple(audio.shape))
        torch.testing.assert_close(audio, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        soundfile_load.assert_called_once_with("sample.wav")
        ffmpeg_load.assert_not_called()

    def test_load_audio_stereo_falls_back_to_ffmpeg(self) -> None:
        """Formats unsupported by soundfile should be decoded by ffmpeg."""

        stereo_audio = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        with patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_soundfile",
            return_value=None,
        ) as soundfile_load, patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_ffmpeg",
            return_value=(stereo_audio, 48000),
        ) as ffmpeg_load:
            audio, sample_rate = load_audio_stereo("sample.mp3", 48000, 240.0)

        self.assertEqual(48000, sample_rate)
        torch.testing.assert_close(audio, stereo_audio)
        soundfile_load.assert_called_once_with("sample.mp3")
        ffmpeg_load.assert_called_once_with("sample.mp3", 48000, 240.0)

    def test_load_audio_stereo_resamples_before_trimming(self) -> None:
        """Resampling should still happen when soundfile returns another rate."""

        def fake_resample(audio: torch.Tensor) -> torch.Tensor:
            return audio.repeat_interleave(2, dim=1)

        stereo_audio = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        resample_path = (
            "acestep.training.dataset_builder_modules.preprocess_audio."
            "torchaudio.transforms.Resample"
        )
        with patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_soundfile",
            return_value=(stereo_audio, 2),
        ), patch(resample_path, return_value=fake_resample) as resample:
            audio, sample_rate = load_audio_stereo("sample.wav", 4, 0.75)

        self.assertEqual(4, sample_rate)
        resample.assert_called_once_with(2, 4)
        torch.testing.assert_close(audio, torch.tensor([[0.0, 0.0, 1.0], [2.0, 2.0, 3.0]]))

    def test_load_audio_stereo_raises_when_all_decoders_fail(self) -> None:
        """An unreadable file should fail with a concise preprocessing error."""

        with patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_soundfile",
            return_value=None,
        ), patch(
            "acestep.training.dataset_builder_modules.preprocess_audio._load_with_ffmpeg",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not load audio"):
                load_audio_stereo("broken.mp3", 48000, 240.0)

    def test_ffmpeg_loader_decodes_float32_stdout(self) -> None:
        """The ffmpeg fallback should convert interleaved PCM to channel-first tensors."""

        interleaved = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        completed = SimpleNamespace(returncode=0, stderr=b"", stdout=interleaved.tobytes())

        with patch.object(preprocess_audio.subprocess, "run", return_value=completed) as run:
            loaded = preprocess_audio._load_with_ffmpeg("sample.mp3", 48000, 1.0)

        self.assertIsNotNone(loaded)
        audio, sample_rate = loaded
        self.assertEqual(48000, sample_rate)
        torch.testing.assert_close(audio, torch.tensor([[0.0, 0.5], [0.25, 0.75]]))
        command = run.call_args.args[0]
        self.assertIn("ffmpeg", command[0])
        self.assertIn("-t", command)
        self.assertIn("1.0", command)
        self.assertIn("pipe:1", command)


if __name__ == "__main__":
    unittest.main()
