"""Tests for training configuration defaults."""

from __future__ import annotations

import unittest

from acestep.training.configs import LoRAConfig, TrainingConfig


class TrainingConfigDefaultsTests(unittest.TestCase):
    """Verify adapter and training defaults used when callers omit optional fields."""

    def test_lora_config_defaults_keep_alpha_and_disable_dropout(self) -> None:
        """Default LoRA configs should use alpha 128 and dropout 0."""

        config = LoRAConfig()

        self.assertEqual(128, config.alpha)
        self.assertEqual(0.0, config.dropout)
        self.assertEqual(128, config.to_dict()["lora_alpha"])
        self.assertEqual(0.0, config.to_dict()["lora_dropout"])

    def test_save_best_defaults_to_enabled(self) -> None:
        """Best-checkpoint tracking should be enabled by default."""

        config = TrainingConfig()

        self.assertTrue(config.save_best)
        self.assertEqual(10, config.save_best_after)
        self.assertEqual(5, config.save_best_smoothing_window)
        self.assertEqual(0.001, config.save_best_min_delta)
        self.assertEqual("constant", config.scheduler_type)


if __name__ == "__main__":
    unittest.main()
