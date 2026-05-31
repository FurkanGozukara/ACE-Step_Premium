"""Tests for generated-song audio post-processing integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from acestep.audio_processing.generated_postprocess import postprocess_generated_sample
from acestep.audio_processing.settings import AudioProcessingSettings


class GeneratedPostprocessTests(unittest.TestCase):
    """Verify generated-song originals and processed outputs are handled correctly."""

    def test_preserve_original_keeps_source_and_writes_processed_file(self) -> None:
        """Default generated post-processing should keep original audio."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.flac"
            _write_test_audio(source)
            settings = AudioProcessingSettings(enabled=True, preserve_original=True)

            result = postprocess_generated_sample(
                source_audio_path=str(source),
                run_dir=temp_dir,
                key="sample",
                settings=settings,
                original_audio_paths={"flac": str(source)},
            )

            self.assertTrue(result["applied"])
            self.assertTrue(source.exists())
            self.assertTrue(Path(result["audio_path"]).is_file())
            self.assertTrue(Path(result["metadata_path"]).is_file())

    def test_unchecked_preserve_original_removes_source_after_success(self) -> None:
        """When preserve is off, the original generated audio is removed after processing."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.flac"
            _write_test_audio(source)
            settings = AudioProcessingSettings(enabled=True, preserve_original=False)

            result = postprocess_generated_sample(
                source_audio_path=str(source),
                run_dir=temp_dir,
                key="sample",
                settings=settings,
                original_audio_paths={"flac": str(source)},
            )

            self.assertTrue(result["applied"])
            self.assertFalse(source.exists())
            self.assertTrue(Path(result["audio_path"]).is_file())


def _write_test_audio(path: Path, sample_rate: int = 48000) -> None:
    """Write deterministic stereo audio."""

    time = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    audio = np.stack(
        [
            0.2 * np.sin(2 * np.pi * 220 * time),
            0.2 * np.sin(2 * np.pi * 440 * time),
        ],
        axis=1,
    ).astype(np.float32)
    sf.write(path, audio, sample_rate)


if __name__ == "__main__":
    unittest.main()
