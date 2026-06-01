"""Tests for the ACE-Step audio-processing DSP pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from acestep.audio_processing.auto_editor_trim import SilenceTrimResult
from acestep.audio_processing.pipeline import process_audio_array
from acestep.audio_processing.presets import DEFAULT_STAGE_VALUES, STAGE_KEYS
from acestep.audio_processing.settings import AudioProcessingSettings


class AudioProcessingPipelineTests(unittest.TestCase):
    """Verify real processing behavior on deterministic generated audio."""

    def test_full_chain_processes_audio_and_targets_lufs(self) -> None:
        """The complete chain should produce finite stereo audio near the LUFS target."""

        sample_rate = 48000
        audio = _test_tone(sample_rate, seconds=1.2)
        values = dict(DEFAULT_STAGE_VALUES)
        values["lufs"] = -16.0
        settings = AudioProcessingSettings(
            enabled=True,
            output_format="wav",
            values=values,
            stages_enabled={key: True for key in STAGE_KEYS},
        )

        result = process_audio_array(audio, sample_rate, settings)

        self.assertEqual(audio.shape, result.after.shape)
        self.assertTrue(np.isfinite(result.after).all())
        self.assertLessEqual(float(np.max(np.abs(result.after))), 1.0)
        self.assertAlmostEqual(-16.0, result.lufs_after, delta=1.5)

    def test_disabled_stages_preserve_audio_samples(self) -> None:
        """Disabled stages should leave samples unchanged apart from mono-to-stereo prep."""

        sample_rate = 24000
        audio = _test_tone(sample_rate, seconds=0.5)
        settings = AudioProcessingSettings(
            stages_enabled={key: False for key in STAGE_KEYS},
        )

        result = process_audio_array(audio, sample_rate, settings)

        np.testing.assert_allclose(audio, result.after, atol=1e-6)

    def test_trim_silent_edges_shortens_processed_audio(self) -> None:
        """Optional silence trimming should remove quiet leading and trailing edges."""

        sample_rate = 1000
        silence = np.zeros((100, 2), dtype=np.float32)
        tone = np.full((300, 2), 0.2, dtype=np.float32)
        audio = np.concatenate([silence, tone, silence], axis=0)
        settings = AudioProcessingSettings(
            trim_empty_output=True,
            trim_threshold_db=-40.0,
            stages_enabled={key: False for key in STAGE_KEYS},
        )

        metadata = {
            "enabled": True,
            "applied": True,
            "reason": "auto_editor_trimmed",
            "trimmed_duration_seconds": 0.3,
        }
        trimmed_tensor = torch.from_numpy(tone.T)
        with patch(
            "acestep.audio_processing.pipeline.trim_silent_edges",
            return_value=SilenceTrimResult(trimmed_tensor, metadata),
        ):
            result = process_audio_array(audio, sample_rate, settings)

        self.assertEqual((300, 2), result.after.shape)
        self.assertTrue(result.trim_metadata["applied"])
        self.assertEqual("auto_editor_trimmed", result.trim_metadata["reason"])


def _test_tone(sample_rate: int, seconds: float) -> np.ndarray:
    """Return deterministic stereo test audio."""

    time = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    left = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
    right = 0.18 * np.sin(2.0 * np.pi * 330.0 * time + 0.25)
    return np.stack([left, right], axis=1).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
