"""Tests for LoRA training handler setup."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.training.lora_training import (
    _checkpoint_epoch_from_name,
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
                        lora_name="test-lora",
                        model_config="model-b",
                    )
                )[0]

        ensure_dit_ready.assert_called_once_with(handler, config_path="model-b")
        self.assertIn("cannot load", first_status)

    def test_start_training_requires_valid_lora_name(self) -> None:
        """Training should stop early when the LoRA name cannot be used as a filename."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            first_status = next(
                start_training(
                    tmpdir,
                    SimpleNamespace(model=None),
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
                    lora_name="bad/name",
                )
            )[0]

        self.assertIn("Invalid LoRA training name", first_status)

    def test_checkpoint_epoch_parses_old_and_named_folders(self) -> None:
        """Export fallback should handle old and new checkpoint folder names."""

        self.assertEqual(10, _checkpoint_epoch_from_name("epoch_10_loss_0.1234"))
        self.assertEqual(32, _checkpoint_epoch_from_name("my awesome-song-32"))
        self.assertEqual(32, _checkpoint_epoch_from_name("my awesome-song-32-sample"))
        self.assertIsNone(_checkpoint_epoch_from_name("best"))

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
                lora_name="my awesome-song",
            )
            _save_training_config_snapshot(LoRAConfig(r=16), training_config)
            loaded = TrainingConfig.from_json(f"{tmpdir}/training_config.json")

        self.assertTrue(loaded.use_fp8)
        self.assertEqual(10, loaded.sample_every_n_epochs)
        self.assertEqual("style", loaded.sample_prompt)
        self.assertEqual("my awesome-song", loaded.lora_name)

    def test_training_config_snapshot_accepts_quoted_output_path_with_spaces(self) -> None:
        """Quoted output paths with spaces should write to the intended directory."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            output_dir = os.path.join(tmpdir, "lora output")
            training_config = TrainingConfig(
                output_dir=f'"{output_dir.replace(os.sep, "/")}"',
            )

            _save_training_config_snapshot(LoRAConfig(r=16), training_config)

            config_path = os.path.join(output_dir, "training_config.json")
            self.assertTrue(os.path.isfile(config_path))
            self.assertEqual(os.path.realpath(output_dir), training_config.output_dir)


if __name__ == "__main__":
    unittest.main()
