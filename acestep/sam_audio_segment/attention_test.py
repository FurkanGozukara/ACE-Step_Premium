"""Tests for SAM-Audio SDPA backend selection."""

import unittest

import torch

from acestep.sam_audio_segment.attention import (
    SAM_ATTENTION_AUTO,
    SAM_ATTENTION_EFFICIENT,
    normalize_attention_backend,
    sdpa_backend_context,
)


class TestSamAudioAttention(unittest.TestCase):
    """Verify attention backend normalization and safe CPU behavior."""

    def test_unknown_backend_defaults_to_auto(self):
        """Unknown saved values should load the official PyTorch SDPA mode."""

        self.assertEqual(SAM_ATTENTION_AUTO, normalize_attention_backend("bad"))

    def test_auto_backend_is_allowed(self):
        """Auto should preserve PyTorch's default SDPA selection."""

        self.assertEqual(SAM_ATTENTION_AUTO, normalize_attention_backend("auto"))

    def test_cpu_context_is_noop(self):
        """CPU processing should not require CUDA SDPA backends."""

        with sdpa_backend_context(SAM_ATTENTION_EFFICIENT, torch.device("cpu")):
            value = torch.tensor([1.0])

        self.assertEqual(1.0, float(value.item()))


if __name__ == "__main__":
    unittest.main()
