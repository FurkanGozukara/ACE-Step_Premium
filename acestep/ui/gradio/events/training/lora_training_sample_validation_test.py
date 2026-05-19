"""Tests for LoRA training sample-generation validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.training.lora_training import start_training


class LoraTrainingSampleValidationTests(unittest.TestCase):
    """Verify sample generation refuses mismatched base models."""

    def test_sample_generation_model_mismatch_stops_before_training(self) -> None:
        """Training should not start when sample settings use another base model."""

        repo_root = Path.cwd()
        with tempfile.TemporaryDirectory(dir=repo_root) as tmpdir:
            tensor_dir = Path(tmpdir) / "tensors"
            output_dir = Path(tmpdir) / "out"
            tensor_dir.mkdir()
            result = next(
                start_training(
                    tensor_dir=str(tensor_dir),
                    dit_handler=None,
                    lora_rank=64,
                    lora_alpha=128,
                    lora_dropout=0.1,
                    learning_rate=1e-4,
                    train_epochs=1,
                    train_batch_size=1,
                    gradient_accumulation=1,
                    save_every_n_epochs=1,
                    training_shift=3.0,
                    training_seed=42,
                    lora_output_dir=str(output_dir),
                    resume_checkpoint_dir="",
                    training_state={"is_training": False, "should_stop": False},
                    lora_name="test_lora",
                    sample_generation_enabled=True,
                    sample_generation_model_config="ACEStep_1_5_XL_Base_BF16",
                    sample_generation_settings={
                        "config_path": "ACEStep_1_5_XL_Base_BF16"
                    },
                    model_config="ACEStep_1_5_XL_Turbo_BF16",
                )
            )

        self.assertIn("does not match", result[0])
        self.assertFalse(result[3]["is_training"])


if __name__ == "__main__":
    unittest.main()
