"""Tests for in-process LoRA sample generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from acestep.training.sample_generation_inprocess import run_training_sample_inprocess


class SampleGenerationInprocessTests(unittest.TestCase):
    """Verify training samples reuse the active handler and Advanced settings."""

    def test_uses_handler_without_child_process_and_forwards_settings(self) -> None:
        """The helper should call inference generation directly with UI settings."""

        handler = SimpleNamespace(
            offload_to_cpu=False,
            offload_dit_to_cpu=True,
            model=SimpleNamespace(decoder=nn.Linear(1, 1)),
        )
        captured = {}

        def fake_generate_music(dit_handler, llm_handler, *, params, config, save_dir):
            generated_path = Path(save_dir) / "generated.flac"
            generated_path.write_bytes(b"audio")
            captured.update(
                {
                    "handler": dit_handler,
                    "llm_handler": llm_handler,
                    "params": params,
                    "config": config,
                    "save_dir": save_dir,
                }
            )
            return SimpleNamespace(
                success=True,
                error=None,
                status_message="ok",
                audios=[
                    {
                        "path": str(generated_path),
                        "key": "abc",
                        "sample_rate": 48000,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("acestep.inference.generate_music", side_effect=fake_generate_music):
                result = run_training_sample_inprocess(
                    handler=handler,
                    output_dir=tmpdir,
                    artifact_basename="test_lora_7",
                    prompt="style",
                    lyrics="lyrics",
                    generation_settings={
                        "audio_duration": 45,
                        "inference_steps": 6,
                        "guidance_scale": 1.25,
                        "audio_format": "mp3",
                        "random_seed_checkbox": False,
                        "seed": 123,
                    },
                    fallback_duration=30,
                    fallback_inference_steps=8,
                    fallback_seed=42,
                    offload_generation=True,
                )
            expected_audio_path = Path(tmpdir) / "test_lora_7.flac"
            expected_metadata_path = Path(tmpdir) / "test_lora_7.json"
            sample_result_path = Path(tmpdir) / "sample_result.json"
            audio_exists = expected_audio_path.is_file()
            metadata_exists = expected_metadata_path.is_file()
            sample_result_exists = sample_result_path.exists()

        self.assertTrue(result["success"])
        self.assertIs(handler, captured["handler"])
        self.assertIsNone(captured["llm_handler"])
        self.assertEqual(45, captured["params"].duration)
        self.assertEqual(6, captured["params"].inference_steps)
        self.assertEqual("style", captured["params"].caption)
        self.assertEqual("lyrics", captured["params"].lyrics)
        self.assertEqual(1.25, captured["params"].guidance_scale)
        self.assertEqual("flac", captured["config"].audio_format)
        self.assertEqual(str(expected_audio_path), result["audios"][0]["path"])
        self.assertTrue(audio_exists)
        self.assertTrue(metadata_exists)
        self.assertFalse(sample_result_exists)
        self.assertFalse(handler.offload_to_cpu)
        self.assertTrue(handler.offload_dit_to_cpu)


if __name__ == "__main__":
    unittest.main()
