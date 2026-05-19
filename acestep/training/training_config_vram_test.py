"""Tests for LoRA training VRAM configuration fields."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.training.configs import TrainingConfig


class TrainingConfigVramTests(unittest.TestCase):
    """Verify VRAM settings are serialized and loaded."""

    def test_vram_fields_round_trip_json(self) -> None:
        """Training config JSON should preserve VRAM and sample settings."""

        config = TrainingConfig(
            use_fp8=True,
            gradient_checkpointing=False,
            activation_cpu_offload=True,
            offload_non_decoder=False,
            keep_frozen_base_in_compute_dtype=False,
            use_8bit_adam=False,
            empty_cache_every_n_steps=3,
            sample_every_n_epochs=5,
            sample_prompt="style",
            sample_lyrics="lyrics",
            sample_duration=25.0,
            sample_inference_steps=4,
            sample_seed=123,
            sample_output_dir="samples",
            sample_offload_training_model=False,
            sample_offload_generation=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "training_config.json"
            config.save_json(path)
            loaded = TrainingConfig.from_json(path)

        self.assertTrue(loaded.use_fp8)
        self.assertFalse(loaded.gradient_checkpointing)
        self.assertTrue(loaded.activation_cpu_offload)
        self.assertFalse(loaded.offload_non_decoder)
        self.assertFalse(loaded.keep_frozen_base_in_compute_dtype)
        self.assertFalse(loaded.use_8bit_adam)
        self.assertEqual(3, loaded.empty_cache_every_n_steps)
        self.assertEqual(5, loaded.sample_every_n_epochs)
        self.assertEqual("style", loaded.sample_prompt)
        self.assertEqual("lyrics", loaded.sample_lyrics)
        self.assertEqual(25.0, loaded.sample_duration)
        self.assertEqual(4, loaded.sample_inference_steps)
        self.assertEqual(123, loaded.sample_seed)
        self.assertEqual("samples", loaded.sample_output_dir)
        self.assertFalse(loaded.sample_offload_training_model)
        self.assertFalse(loaded.sample_offload_generation)


if __name__ == "__main__":
    unittest.main()
