"""Tests for audio-code parsing and latent decode helpers."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vector_quantize_pytorch import ResidualFSQ

from acestep.core.generation.handler.audio_codes import AudioCodesMixin


class _Host(AudioCodesMixin):
    """Minimal host exposing audio-code decode dependencies."""

    def __init__(self, quantizer: ResidualFSQ, dtype: torch.dtype = torch.bfloat16):
        """Initialize the fake handler state for decode tests."""
        self.device = torch.device("cpu")
        self.dtype = dtype
        self.model = SimpleNamespace(
            tokenizer=SimpleNamespace(quantizer=quantizer),
            detokenizer=torch.nn.Identity(),
        )

    def _load_model_context(self, _name: str):
        """Return a no-op model context manager."""
        return nullcontext()


def _make_mismatched_bf16_quantizer() -> ResidualFSQ:
    """Return a small ResidualFSQ with BF16 projection and float32 scales."""
    quantizer = ResidualFSQ(levels=[8, 8], num_quantizers=1, dim=4)
    quantizer.project_out.to(dtype=torch.bfloat16)
    for layer in quantizer.layers:
        layer.implicit_codebook = layer.implicit_codebook.to(torch.bfloat16)
    quantizer.scales = quantizer.scales.to(torch.float32)
    return quantizer


class AudioCodesMixinTests(unittest.TestCase):
    """Verify semantic audio-code conversion behavior."""

    def test_decode_casts_residual_fsq_scales_before_projection(self) -> None:
        """BF16 code decode should not fail when ResidualFSQ scales start float32."""
        quantizer = _make_mismatched_bf16_quantizer()
        host = _Host(quantizer)

        result = host._decode_audio_codes_to_latents(
            "<|audio_code_1|><|audio_code_2|><|audio_code_3|>"
        )

        self.assertIsNotNone(result)
        self.assertEqual(torch.bfloat16, result.dtype)
        self.assertEqual(torch.bfloat16, quantizer.scales.dtype)

    def test_decode_returns_none_for_empty_audio_codes(self) -> None:
        """Empty audio-code strings should still bypass quantizer decoding."""
        quantizer = _make_mismatched_bf16_quantizer()
        host = _Host(quantizer)

        result = host._decode_audio_codes_to_latents("")

        self.assertIsNone(result)
        self.assertEqual(torch.float32, quantizer.scales.dtype)


if __name__ == "__main__":
    unittest.main()
