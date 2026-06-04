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

    def test_safetensors_loader_skips_prefixes_before_loading_tensors(self):
        """Safetensors filtering should avoid materializing skipped keys."""

        fake_checkpoint = _FakeSafeOpen(
            {
                "keep.weight": torch.ones(1),
                "vision_encoder.weight": torch.zeros(1),
            }
        )
        with patch(
            "acestep.sam_audio_segment.runtime.safe_open",
            return_value=fake_checkpoint,
        ) as safe_open:
            result = load_checkpoint(
                Path("model.safetensors"),
                device="cuda",
                skip_prefixes=("vision_encoder.",),
            )

        self.assertEqual(["keep.weight"], list(result))
        self.assertEqual(["keep.weight"], fake_checkpoint.loaded_keys)
        safe_open.assert_called_once_with("model.safetensors", framework="pt", device="cuda")


class _FakeSafeOpen:
    """Small ``safe_open`` double that records materialized tensors."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors
        self.loaded_keys: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def keys(self):
        """Return available checkpoint keys."""

        return self.tensors.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        """Return one tensor and record that it was materialized."""

        self.loaded_keys.append(key)
        return self.tensors[key]


if __name__ == "__main__":
    unittest.main()
