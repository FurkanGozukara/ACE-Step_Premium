"""Tests for training configuration defaults."""

from __future__ import annotations

import unittest

from acestep.training.configs import LoRAConfig


class TrainingConfigDefaultsTests(unittest.TestCase):
    """Verify adapter default configs stay aligned with UI defaults."""

    def test_lora_config_defaults_keep_alpha_and_disable_dropout(self) -> None:
        """Default LoRA configs should use alpha 128 and dropout 0."""

        config = LoRAConfig()

        self.assertEqual(128, config.alpha)
        self.assertEqual(0.0, config.dropout)
        self.assertEqual(128, config.to_dict()["lora_alpha"])
        self.assertEqual(0.0, config.to_dict()["lora_dropout"])


if __name__ == "__main__":
    unittest.main()
