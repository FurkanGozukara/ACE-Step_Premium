"""Tests for scaled FP8 DiT quantization helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from acestep.core.generation.handler.fp8_scaled_quantization import (
    apply_fp8_scaled_quantization,
)


class _TinyDit(nn.Module):
    """Small DiT-like module with decoder and non-decoder linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.block = nn.Linear(64, 4, bias=False)
        self.encoder = nn.Linear(64, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run decoder block for output comparison."""

        return self.decoder.block(x)


class Fp8ScaledQuantizationTests(unittest.TestCase):
    """Verify scaled FP8 quantization patches and caches decoder linears."""

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch float8 required")
    def test_quantizes_decoder_linear_and_writes_cache(self) -> None:
        """It quantizes only decoder Linear weights and keeps forward usable."""

        with tempfile.TemporaryDirectory() as tmp:
            old_cache = os.environ.get("ACESTEP_FP8_CACHE_DIR")
            os.environ["ACESTEP_FP8_CACHE_DIR"] = tmp
            try:
                model = _TinyDit()
                source = model(torch.ones(1, 64))

                apply_fp8_scaled_quantization(model, checkpoint_path=tmp, device="cpu")
                result = model(torch.ones(1, 64))

                self.assertEqual(model.decoder.block.weight.dtype, torch.float8_e4m3fn)
                self.assertTrue(hasattr(model.decoder.block, "scale_weight"))
                self.assertNotEqual(model.encoder.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(result.shape, source.shape)
                self.assertTrue(list(os.scandir(tmp)))
            finally:
                if old_cache is None:
                    os.environ.pop("ACESTEP_FP8_CACHE_DIR", None)
                else:
                    os.environ["ACESTEP_FP8_CACHE_DIR"] = old_cache


if __name__ == "__main__":
    unittest.main()
