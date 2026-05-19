"""Tests for LoRA training VRAM optimization helpers."""

from __future__ import annotations

import unittest

import torch.nn as nn

from acestep.training.vram_optimizations import offload_handler_training_modules


class _Handler:
    """Small handler stand-in with inference modules."""

    def __init__(self) -> None:
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.tokenizer = object()


class VramOptimizationsTests(unittest.TestCase):
    """Verify training memory helpers only move intended modules."""

    def test_offload_handler_training_modules_moves_inference_modules(self) -> None:
        """Handler VAE and text encoder are offloaded for tensor LoRA training."""

        handler = _Handler()

        moved = offload_handler_training_modules(handler)

        self.assertEqual(moved, ("vae", "text_encoder"))
        self.assertEqual(next(handler.vae.parameters()).device.type, "cpu")
        self.assertEqual(next(handler.text_encoder.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
