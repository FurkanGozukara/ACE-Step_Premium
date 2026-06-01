"""Tests for SAM-Audio media output saving."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import torch

from acestep.audio_processing.auto_editor_trim import SilenceTrimResult
from acestep.sam_audio_segment.media_output import save_sam_audio_outputs


class TestSamAudioMediaOutput(unittest.TestCase):
    """Verify saved SAM-Audio artifact behavior."""

    def test_save_outputs_trims_target_audio_when_enabled(self) -> None:
        """Saving with trim enabled writes the shortened target and metadata."""

        sample_rate = 48000
        target = torch.zeros(1, sample_rate * 3)
        target[:, sample_rate : sample_rate * 2] = 0.25
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trimmed = target[:, sample_rate : sample_rate * 2]
            metadata = {"enabled": True, "applied": True, "reason": "auto_editor_trimmed"}
            with patch(
                "acestep.sam_audio_segment.media_output.trim_silent_edges",
                return_value=SilenceTrimResult(trimmed, metadata),
            ):
                artifacts = save_sam_audio_outputs(
                    source_path=root / "source.wav",
                    output_dir=root / "out",
                    output_stem="clip",
                    target=target,
                    residual=None,
                    sample_rate=sample_rate,
                    output_format="wav",
                    include_residual=False,
                    include_video=False,
                    metadata={"settings": {}},
                    trim_empty_output=True,
                    trim_threshold_db=-40.0,
                )

            audio_info = sf.info(artifacts.target_audio_path)
            saved_metadata = json.loads(Path(artifacts.metadata_path).read_text())

        self.assertEqual(sample_rate, audio_info.frames)
        self.assertAlmostEqual(1.0, artifacts.duration_seconds, places=3)
        self.assertTrue(saved_metadata["trim"]["applied"])
        self.assertEqual("auto_editor_trimmed", saved_metadata["trim"]["reason"])
        self.assertEqual(1.0, saved_metadata["metrics"]["duration_seconds"])


if __name__ == "__main__":
    unittest.main()
