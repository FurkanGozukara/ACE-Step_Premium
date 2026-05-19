"""Tests for LoRA training handler setup."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.ui.gradio.events.training.lora_training import (
    _save_training_config_snapshot,
    _uses_fp8_scaled,
    start_training,
)


class LoRATrainingHandlerTests(unittest.TestCase):
    """Verify selected base model initialization before LoRA training."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_start_training_requests_selected_model_before_training(self) -> None:
        """A selected base model should be passed into DiT readiness checks."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            handler = SimpleNamespace(model=None)
            with patch(
                "acestep.ui.gradio.events.training.lora_training.ensure_dit_ready",
                return_value=(False, "cannot load"),
            ) as ensure_dit_ready:
                first_status = next(
                    start_training(
                        tmpdir,
                        handler,
                        64,
                        128,
                        0.1,
                        0.0003,
                        10,
                        1,
                        1,
                        10,
                        3.0,
                        42,
                        "out",
                        "",
                        {},
                        model_config="model-b",
                    )
                )[0]

        ensure_dit_ready.assert_called_once_with(handler, config_path="model-b")
        self.assertIn("cannot load", first_status)

    def test_uses_fp8_scaled_normalizes_dropdown_label(self) -> None:
        """Scaled FP8 should be enabled only by the matching dropdown value."""

        self.assertTrue(_uses_fp8_scaled("FP8 scaled"))
        self.assertTrue(_uses_fp8_scaled(" fp8 SCALED "))
        self.assertFalse(_uses_fp8_scaled("Disabled"))

    def test_training_config_snapshot_writes_loadable_training_config(self) -> None:
        """Saved Gradio training config should be loadable by TrainingConfig."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            training_config = TrainingConfig(
                output_dir=tmpdir,
                use_fp8=True,
                sample_every_n_epochs=10,
                sample_prompt="style",
            )
            _save_training_config_snapshot(LoRAConfig(r=16), training_config)
            loaded = TrainingConfig.from_json(f"{tmpdir}/training_config.json")

        self.assertTrue(loaded.use_fp8)
        self.assertEqual(10, loaded.sample_every_n_epochs)
        self.assertEqual("style", loaded.sample_prompt)


if __name__ == "__main__":
    unittest.main()
