"""Regression tests for nano-vLLM rotary embedding configuration handling."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from nanovllm.layers import rotary_embedding
from nanovllm.models import qwen3


class GetRopeTests(unittest.TestCase):
    """Verify legacy and Transformers 5 RoPE schemas share the cached path."""

    def setUp(self):
        rotary_embedding._get_rope_cached.cache_clear()

    def tearDown(self):
        rotary_embedding._get_rope_cached.cache_clear()

    def test_transformers_5_default_rope_mapping_is_normalized_before_cache(self):
        marker = MagicMock()
        marker.cos_sin_cache.device = torch.device("cpu")
        rope_parameters = {
            "rope_theta": 1_000_000,
            "rope_type": "default",
        }

        with patch.object(
            rotary_embedding,
            "RotaryEmbedding",
            return_value=marker,
        ) as rotary_embedding_class:
            first = rotary_embedding.get_rope(
                128,
                rotary_dim=128,
                max_position=40960,
                base=10000,
                rope_scaling=rope_parameters,
                device="cpu",
            )
            second = rotary_embedding.get_rope(
                128,
                rotary_dim=128,
                max_position=40960,
                base=1_000_000,
                rope_scaling=None,
                device=torch.device("cpu"),
            )

        self.assertIs(first, marker)
        self.assertIs(second, marker)
        rotary_embedding_class.assert_called_once_with(
            128,
            128,
            40960,
            1_000_000.0,
            device=torch.device("cpu"),
        )

    def test_legacy_default_type_alias_is_supported(self):
        marker = MagicMock()
        marker.cos_sin_cache.device = torch.device("cpu")
        with patch.object(
            rotary_embedding,
            "RotaryEmbedding",
            return_value=marker,
        ) as rotary_embedding_class:
            rotary_embedding.get_rope(
                64,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                rope_scaling={"type": "default", "rope_theta": 500000},
                device="cpu",
            )

        rotary_embedding_class.assert_called_once_with(
            64,
            64,
            2048,
            500000.0,
            device=torch.device("cpu"),
        )

    def test_unsupported_scaling_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "only supports default RoPE"):
            rotary_embedding.get_rope(
                128,
                rotary_dim=128,
                max_position=40960,
                base=1_000_000,
                rope_scaling={"rope_type": "linear", "factor": 2.0},
                device="cpu",
            )

    def test_partial_rotary_embedding_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "full-head rotary embeddings"):
            rotary_embedding.get_rope(
                128,
                rotary_dim=128,
                max_position=40960,
                base=1_000_000,
                rope_scaling={
                    "rope_type": "default",
                    "partial_rotary_factor": 0.5,
                },
                device="cpu",
            )

    def test_cache_key_includes_resolved_device(self):
        cpu_marker = MagicMock()
        meta_marker = MagicMock()
        cpu_marker.cos_sin_cache.device = torch.device("cpu")
        meta_marker.cos_sin_cache.device = torch.device("meta")

        with patch.object(
            rotary_embedding,
            "RotaryEmbedding",
            side_effect=[cpu_marker, meta_marker],
        ) as rotary_embedding_class:
            first = rotary_embedding.get_rope(8, 8, 32, 10000, device="cpu")
            second = rotary_embedding.get_rope(8, 8, 32, 10000, device="meta")

        self.assertIs(first, cpu_marker)
        self.assertIs(second, meta_marker)
        self.assertEqual(rotary_embedding_class.call_count, 2)

    def test_stale_cached_module_is_rebuilt_on_requested_device(self):
        first = rotary_embedding.get_rope(8, 8, 32, 10000, device="cpu")
        first.to("meta")

        second = rotary_embedding.get_rope(8, 8, 32, 10000, device="cpu")

        self.assertIsNot(first, second)
        self.assertEqual(second.cos_sin_cache.device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_compiled_forward_keeps_rope_cache_with_cuda_inputs(self):
        device = torch.device("cuda", torch.cuda.current_device())
        rope = rotary_embedding.get_rope(8, 8, 32, 10000, device=device)
        positions = torch.tensor([0, 1], device=device)
        query = torch.randn(2, 1, 8, device=device)
        key = torch.randn(2, 1, 8, device=device)

        rotated_query, rotated_key = rope(positions, query, key)

        self.assertEqual(rope.cos_sin_cache.device, device)
        self.assertEqual(rotated_query.device, device)
        self.assertEqual(rotated_key.device, device)

    def test_qwen_attention_binds_rope_to_projection_device(self):
        marker = MagicMock()
        with patch.object(qwen3, "get_rope", return_value=marker) as get_rope:
            attention = qwen3.Qwen3Attention(
                hidden_size=16,
                num_heads=2,
                num_kv_heads=1,
                max_position=32,
                head_dim=8,
                rope_theta=10000,
                rope_scaling={"rope_type": "default"},
            )

        self.assertEqual(
            get_rope.call_args.kwargs["device"],
            attention.qkv_proj.weight.device,
        )


if __name__ == "__main__":
    unittest.main()
