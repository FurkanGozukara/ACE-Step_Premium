"""Tests for low-level SAM-Audio checkpoint loading helpers."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from acestep.sam_audio_segment.runtime import load_checkpoint


class LoadCheckpointTests(unittest.TestCase):
    """Verify checkpoint loading preserves the requested placement."""

    def test_safetensors_loader_receives_requested_device(self):
        """Safetensors checkpoints should load on the caller's target device."""

        expected = {"weight": torch.zeros(1)}
        with patch(
            "acestep.sam_audio_segment.runtime.load_file",
            return_value=expected,
        ) as load_file:
            result = load_checkpoint(Path("model.safetensors"), device="cuda")

        self.assertIs(result, expected)
        load_file.assert_called_once_with("model.safetensors", device="cuda")


if __name__ == "__main__":
    unittest.main()
