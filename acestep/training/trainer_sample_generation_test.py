"""Tests for LoRA trainer checkpoint sample output paths."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training.configs import TrainingConfig
from acestep.training.trainer import (
    LoRATrainer,
    _save_final_lora_artifacts,
    _should_save_epoch_checkpoint,
)


class LoRATrainerSampleGenerationTests(unittest.TestCase):
    """Verify checkpoint samples are saved under the named run folder."""

    def test_zero_checkpoint_interval_disables_periodic_saves(self) -> None:
        """Save Every N Epochs = 0 should not trigger modulo checkpoint logic."""

        config = TrainingConfig(save_every_n_epochs=0)

        self.assertFalse(_should_save_epoch_checkpoint(config, 1))
        self.assertFalse(_should_save_epoch_checkpoint(config, 100))

    def test_final_lora_artifacts_always_save_flat_checkpoint(self) -> None:
        """Final save should write a flat final safetensors and resume state."""

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                output_dir=tmpdir,
                lora_name="2pac_SFT-3e-04",
                max_epochs=3,
                save_every_n_epochs=0,
            )
            optimizer = SimpleNamespace()
            scheduler = SimpleNamespace()

            with patch(
                "acestep.training.trainer.save_training_checkpoint",
            ) as save_checkpoint:
                final_path = _save_final_lora_artifacts(
                    SimpleNamespace(),
                    optimizer,
                    scheduler,
                    config,
                    global_step=12,
                )

        self.assertEqual(tmpdir, final_path)
        save_checkpoint.assert_called_once()
        self.assertEqual(tmpdir, save_checkpoint.call_args.args[5])
        self.assertEqual(
            "2pac_SFT-3e-04-epoch-3-final",
            save_checkpoint.call_args.kwargs["artifact_name"],
        )
        self.assertEqual("final", save_checkpoint.call_args.kwargs["state_suffix"])

    def test_checkpoint_sample_uses_flat_named_samples_folder(self) -> None:
        """A checkpoint sample should use ``samples/<lora_name>_<epoch>`` names."""

        captured = {}

        def fake_sample(**kwargs):
            captured.update(kwargs)
            audio_path = os.path.join(
                kwargs["output_dir"],
                f"{kwargs['artifact_basename']}.flac",
            )
            return {
                "success": True,
                "audios": [{"path": audio_path}],
                "peak_vram_gb": 0.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LoRATrainer.__new__(LoRATrainer)
            trainer.training_config = TrainingConfig(
                output_dir=tmpdir,
                lora_name="2pac_SFT-3e-04",
                sample_every_n_epochs=1,
                sample_prompt="style",
                sample_lyrics="lyrics",
            )
            trainer.module = SimpleNamespace(device="cpu")
            trainer.dit_handler = object()

            with patch(
                "acestep.training.trainer.sample_generation_vram_guard",
                return_value=nullcontext(),
            ), patch(
                "acestep.training.trainer.run_training_sample_inprocess",
                side_effect=fake_sample,
            ):
                status = trainer._generate_checkpoint_sample("checkpoint", 3)

        self.assertEqual(os.path.join(tmpdir, "samples"), captured["output_dir"])
        self.assertEqual("2pac_SFT-3e-04_3", captured["artifact_basename"])
        self.assertIn("2pac_SFT-3e-04_3.flac", status)


if __name__ == "__main__":
    unittest.main()
