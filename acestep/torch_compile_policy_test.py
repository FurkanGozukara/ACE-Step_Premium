"""Tests for the measured ACE-Step inference compile policy."""

from __future__ import annotations

import unittest

from acestep.torch_compile_policy import (
    DIT_EAGER_DETAIL,
    VAE_EAGER_DETAIL,
    resolve_inference_compile_policy,
)


class InferenceCompilePolicyTests(unittest.TestCase):
    def test_enabled_request_compiles_only_lm(self) -> None:
        policy = resolve_inference_compile_policy(True)

        self.assertTrue(policy.requested)
        self.assertTrue(policy.lm)
        self.assertFalse(policy.dit)
        self.assertFalse(policy.vae)
        self.assertTrue(policy.enabled_for("5hz_lm"))
        self.assertFalse(policy.enabled_for("dit"))
        self.assertFalse(policy.enabled_for("vae"))
        self.assertEqual(DIT_EAGER_DETAIL, policy.disabled_detail("dit"))
        self.assertEqual(VAE_EAGER_DETAIL, policy.disabled_detail("vae"))

    def test_disabled_request_compiles_nothing(self) -> None:
        policy = resolve_inference_compile_policy(False)

        self.assertFalse(policy.lm)
        self.assertFalse(policy.dit)
        self.assertFalse(policy.vae)
        self.assertEqual("disabled by user option", policy.disabled_detail("dit"))

    def test_unknown_component_is_rejected(self) -> None:
        policy = resolve_inference_compile_policy(True)

        with self.assertRaises(ValueError):
            policy.enabled_for("unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
