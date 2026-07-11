"""Tests for the measured SAM-Audio compile policy."""

from __future__ import annotations

import unittest

from .compile_policy import resolve_sam_compile_policy


class SamCompilePolicyTests(unittest.TestCase):
    def test_enabled_request_compiles_only_diffusion_forward(self) -> None:
        policy = resolve_sam_compile_policy(True)

        self.assertTrue(policy.requested)
        self.assertEqual(
            {
                "diffusion_forward": True,
                "codec_encoder": False,
                "codec_decoder": False,
                "text_encoder": False,
                "span_predictor": False,
                "rankers": False,
            },
            policy.targets(),
        )
        self.assertIn("measured slower warm", policy.disabled_detail("codec_encoder"))
        self.assertIn("no measured warm benefit", policy.disabled_detail("text_encoder"))

    def test_disabled_request_compiles_nothing(self) -> None:
        policy = resolve_sam_compile_policy(False)

        self.assertFalse(policy.requested)
        self.assertFalse(any(policy.targets().values()))
        self.assertEqual(
            "disabled by user option",
            policy.disabled_detail("diffusion_forward"),
        )

    def test_unknown_component_is_rejected(self) -> None:
        policy = resolve_sam_compile_policy(True)

        with self.assertRaises(ValueError):
            policy.enabled_for("unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
