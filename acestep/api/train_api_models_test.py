"""Tests for training API request model defaults."""

from __future__ import annotations

import unittest

from acestep.api.train_api_models import StartTrainingRequest


class TrainApiModelDefaultsTests(unittest.TestCase):
    """Verify training API defaults match UI training defaults."""

    def test_lora_defaults_keep_alpha_and_disable_dropout(self) -> None:
        """LoRA API defaults should use alpha 128 and dropout 0."""

        request = StartTrainingRequest(tensor_dir="tensors")

        self.assertEqual(128, request.lora_alpha)
        self.assertEqual(0.0, request.lora_dropout)


if __name__ == "__main__":
    unittest.main()
