"""Tests for ACE-Step Extract remaining-audio helpers."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from acestep.ui.gradio.events.results.extract_remaining_audio import (
    save_extract_remaining_audio,
)


def _absolute_projection(audio: np.ndarray, basis: np.ndarray) -> float:
    """Return the absolute scalar projection of audio onto a basis waveform."""

    return abs(float(np.dot(audio, basis) / np.dot(basis, basis)))


class ExtractRemainingAudioTests(unittest.TestCase):
    """Verify source-minus-extracted audio export behavior."""

    def test_save_extract_remaining_audio_writes_source_minus_extracted(self) -> None:
        """A remaining-audio WAV should contain source audio minus extracted audio."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.wav"
            sample_rate = 8_000
            source = np.array(
                [
                    [0.50, -0.25],
                    [0.25, 0.25],
                    [-0.50, 0.50],
                ],
                dtype=np.float32,
            )
            extracted = torch.tensor(
                [
                    [0.10, 0.15, -0.20],
                    [-0.05, 0.05, 0.25],
                ],
                dtype=torch.float32,
            )
            sf.write(str(source_path), source, sample_rate, format="WAV", subtype="FLOAT")

            metadata = save_extract_remaining_audio(
                source_audio_path=source_path,
                extracted_audio=extracted,
                sample_rate=sample_rate,
                output_dir=root,
                output_stem="extracted",
                output_format="wav",
            )

            remaining_path = Path(metadata["remaining_audio_path"])
            remaining, remaining_sample_rate = sf.read(
                str(remaining_path),
                dtype="float32",
                always_2d=True,
            )

            self.assertTrue(metadata["applied"])
            self.assertEqual(sample_rate, metadata["sample_rate"])
            self.assertEqual(sample_rate, remaining_sample_rate)
            self.assertEqual(root / "extracted_remaining.wav", remaining_path)
            self.assertEqual(
                "waveform_subtraction_short_audio",
                metadata["subtraction"]["method"],
            )
            np.testing.assert_allclose(remaining, source - extracted.numpy().T, atol=1e-4)

    def test_save_extract_remaining_audio_removes_extracted_spectral_energy(self) -> None:
        """Longer remaining-audio export should suppress the extracted stem spectrum."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.wav"
            sample_rate = 8_000
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            extracted_wave = (0.35 * np.sin(2 * math.pi * 440 * time)).astype(np.float32)
            kept_wave = (0.25 * np.sin(2 * math.pi * 880 * time)).astype(np.float32)
            source_wave = (extracted_wave + kept_wave).astype(np.float32)
            source = np.column_stack([source_wave, source_wave]).astype(np.float32)
            extracted = torch.tensor(
                np.stack([extracted_wave, extracted_wave]),
                dtype=torch.float32,
            )
            sf.write(str(source_path), source, sample_rate, format="WAV", subtype="FLOAT")

            metadata = save_extract_remaining_audio(
                source_audio_path=source_path,
                extracted_audio=extracted,
                sample_rate=sample_rate,
                output_dir=root,
                output_stem="tones",
                output_format="wav",
            )

            remaining, _remaining_sample_rate = sf.read(
                metadata["remaining_audio_path"],
                dtype="float32",
                always_2d=True,
            )
            remaining_channel = remaining[:, 0]
            extracted_projection = _absolute_projection(remaining_channel, extracted_wave)
            kept_projection = _absolute_projection(remaining_channel, kept_wave)

            self.assertEqual(
                "source_phase_spectral_subtraction",
                metadata["subtraction"]["method"],
            )
            self.assertLess(extracted_projection, 0.12)
            self.assertGreater(kept_projection, 0.80)

    def test_save_extract_remaining_audio_repeats_mono_extracted_audio(self) -> None:
        """Mono extracted audio should be matched to a stereo source before subtraction."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.wav"
            sample_rate = 8_000
            source = np.array([[0.5, 0.25], [0.25, -0.25]], dtype=np.float32)
            extracted = torch.tensor([[0.10, 0.20]], dtype=torch.float32)
            sf.write(str(source_path), source, sample_rate, format="WAV", subtype="FLOAT")

            metadata = save_extract_remaining_audio(
                source_audio_path=source_path,
                extracted_audio=extracted,
                sample_rate=sample_rate,
                output_dir=root,
                output_stem="mono",
                output_format="wav",
            )

            remaining, _sample_rate = sf.read(
                metadata["remaining_audio_path"],
                dtype="float32",
                always_2d=True,
            )

            expected = source - np.array([[0.10, 0.10], [0.20, 0.20]], dtype=np.float32)
            np.testing.assert_allclose(remaining, expected, atol=1e-4)

    def test_save_extract_remaining_audio_forwards_mp3_quality(self) -> None:
        """MP3 remaining-audio export should receive the configured MP3 quality."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.wav"
            sample_rate = 8_000
            sf.write(
                str(source_path),
                np.zeros((2, 1), dtype=np.float32),
                sample_rate,
                format="WAV",
            )

            with patch(
                "acestep.ui.gradio.events.results.extract_remaining_audio.save_audio",
                return_value=str(root / "stem_remaining.mp3"),
            ) as save_audio_mock:
                metadata = save_extract_remaining_audio(
                    source_audio_path=source_path,
                    extracted_audio=torch.zeros(1, 2),
                    sample_rate=sample_rate,
                    output_dir=root,
                    output_stem="stem",
                    output_format="mp3",
                    mp3_bitrate="256k",
                    mp3_sample_rate=48000,
                )

            self.assertTrue(metadata["applied"])
            self.assertEqual("mp3", metadata["audio_format"])
            self.assertEqual("256k", save_audio_mock.call_args.kwargs["mp3_bitrate"])
            self.assertEqual(48000, save_audio_mock.call_args.kwargs["mp3_sample_rate"])

    def test_save_extract_remaining_audio_skips_missing_source(self) -> None:
        """Missing source audio should return skip metadata instead of writing output."""

        metadata = save_extract_remaining_audio(
            source_audio_path=None,
            extracted_audio=torch.zeros(1, 2),
            sample_rate=8_000,
            output_dir=".",
            output_stem="missing",
            output_format="wav",
        )

        self.assertFalse(metadata["applied"])
        self.assertEqual("missing_source_audio", metadata["error"])


if __name__ == "__main__":
    unittest.main()
