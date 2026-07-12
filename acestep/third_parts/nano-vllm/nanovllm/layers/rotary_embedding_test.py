"""Regression tests for nano-vLLM rotary embedding configuration handling."""

import unittest
from unittest.mock import patch

from nanovllm.layers import rotary_embedding


class GetRopeTests(unittest.TestCase):
    """Verify legacy and Transformers 5 RoPE schemas share the cached path."""

    def setUp(self):
        rotary_embedding._get_rope_cached.cache_clear()

    def tearDown(self):
        rotary_embedding._get_rope_cached.cache_clear()

    def test_transformers_5_default_rope_mapping_is_normalized_before_cache(self):
        marker = object()
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
            )
            second = rotary_embedding.get_rope(
                128,
                rotary_dim=128,
                max_position=40960,
                base=1_000_000,
                rope_scaling=None,
            )

        self.assertIs(first, marker)
        self.assertIs(second, marker)
        rotary_embedding_class.assert_called_once_with(128, 128, 40960, 1_000_000.0)

    def test_legacy_default_type_alias_is_supported(self):
        with patch.object(rotary_embedding, "RotaryEmbedding") as rotary_embedding_class:
            rotary_embedding.get_rope(
                64,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                rope_scaling={"type": "default", "rope_theta": 500000},
            )

        rotary_embedding_class.assert_called_once_with(64, 64, 2048, 500000.0)

    def test_unsupported_scaling_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "only supports default RoPE"):
            rotary_embedding.get_rope(
                128,
                rotary_dim=128,
                max_position=40960,
                base=1_000_000,
                rope_scaling={"rope_type": "linear", "factor": 2.0},
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
            )


if __name__ == "__main__":
    unittest.main()
