"""Tests for LoRA training subprocess helpers."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.training.subprocess_lora_training import (
    _lora_training_dit_init_params,
)


class LoraTrainingSubprocessTests(unittest.TestCase):
    """Verify worker init options derived from LoRA training settings."""

    def test_offload_non_decoder_moves_init_non_decoder_modules_to_cpu(self) -> None:
        """The worker should avoid a GPU spike for modules training will offload."""

        params = {"project_root": "root", "offload_to_cpu": False}

        result = _lora_training_dit_init_params(params, {"offload_non_decoder": True})

        self.assertTrue(result["offload_to_cpu"])
        self.assertFalse(params["offload_to_cpu"])

    def test_without_non_decoder_offload_preserves_init_params(self) -> None:
        """The worker should keep init settings when non-decoder offload is disabled."""

        params = {"project_root": "root", "offload_to_cpu": False}

        result = _lora_training_dit_init_params(params, {"offload_non_decoder": False})

        self.assertFalse(result["offload_to_cpu"])


if __name__ == "__main__":
    unittest.main()
