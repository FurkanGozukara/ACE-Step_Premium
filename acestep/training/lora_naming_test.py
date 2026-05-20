"""Tests for LoRA training artifact naming helpers."""

from __future__ import annotations

import unittest

from acestep.training.lora_naming import (
    lora_epoch_name,
    lora_safetensors_filename,
    lora_training_state_filename,
    validate_lora_name,
)


class LoraNamingTests(unittest.TestCase):
    """Verify user-provided LoRA names are safe and readable."""

    def test_accepts_spaces_and_hyphens(self) -> None:
        """Names with spaces should be preserved for readable artifacts."""

        name, error = validate_lora_name("  my awesome-song  ")

        self.assertIsNone(error)
        self.assertEqual("my awesome-song", name)
        self.assertEqual("my awesome-song-epoch-32", lora_epoch_name(name, 32))
        self.assertEqual(
            "my awesome-song-epoch-32.safetensors",
            lora_safetensors_filename(lora_epoch_name(name, 32)),
        )
        self.assertEqual(
            "epoch-32-training_resume_state.pt",
            lora_training_state_filename(32),
        )
        self.assertEqual(
            "epoch-32-training_resume_state-final.pt",
            lora_training_state_filename(32, suffix="final"),
        )

    def test_rejects_path_separators(self) -> None:
        """Names must not be able to create nested or absolute paths."""

        name, error = validate_lora_name("bad/name")

        self.assertEqual("bad/name", name)
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
