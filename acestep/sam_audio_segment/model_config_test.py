"""Tests for SAM-Audio model configuration helpers."""

import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.sam_audio_segment.model_config import config_for_settings
from acestep.sam_audio_segment.settings import SamAudioSettings


class TestModelConfig(unittest.TestCase):
    """Verify optional ranker config uses local assets."""

    def test_judge_ranker_uses_local_checkpoint(self):
        """Judge ranker config should not reference the online model id."""

        with patch(
            "acestep.sam_audio_segment.model_config.prepare_local_judge_model_dir",
            return_value=Path("models/Sam-Audio-Judge"),
        ), patch(
            "acestep.sam_audio_segment.model_config.resolve_local_judge_checkpoint",
            return_value=Path("models/Sam-Audio-Judge-BF16.safetensors"),
        ), patch(
            "acestep.sam_audio_segment.model_config.normalize_ranker_mode",
            return_value="judge",
        ):
            config = config_for_settings(SamAudioSettings(ranker_mode="judge"))

        self.assertEqual(
            str(Path("models/Sam-Audio-Judge")),
            config["text_ranker"]["checkpoint_or_model_id"],
        )
        self.assertEqual(
            str(Path("models/Sam-Audio-Judge-BF16.safetensors")),
            config["text_ranker"]["checkpoint_path"],
        )
        self.assertNotIn("facebook/sam-audio-judge", str(config["text_ranker"]))


if __name__ == "__main__":
    unittest.main()
