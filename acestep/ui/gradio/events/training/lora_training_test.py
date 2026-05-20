"""Tests for LoRA training handler setup."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.lora_vram_presets import LORA_VRAM_PRESET_24GB_PLUS
from acestep.training.path_safety import (
    discover_default_safe_roots,
    get_safe_roots,
    set_safe_roots,
)
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

    def test_start_training_accepts_default_absolute_tensor_dir(self) -> None:
        """Training should accept absolute tensor folders under default roots."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots(discover_default_safe_roots())
            output_dir = os.path.join(tmpdir, "lora")
            handler = SimpleNamespace(model=None)
            with patch(
                "acestep.ui.gradio.events.training.lora_training.ensure_dit_ready",
                return_value=(False, "cannot load"),
            ):
                first_status = next(
                    start_training(
                        tmpdir,
                        handler,
                        64,
                        128,
                        0.0,
                        0.0003,
                        10,
                        1,
                        1,
                        10,
                        3.0,
                        42,
                        output_dir,
                        "",
                        {},
                        lora_name="test-lora",
                    )
                )[0]

        self.assertNotIn("Rejected unsafe tensor directory path", first_status)
        self.assertIn("cannot load", first_status)

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

    def test_start_training_uses_submitted_values_not_selected_preset(self) -> None:
        """Training configs should use current Gradio values after a preset edit."""

        captured = {}

        class FakeTrainer:
            """Capture trainer configs without running a real training loop."""

            def __init__(self, dit_handler, lora_config, training_config) -> None:
                self.dit_handler = dit_handler
                captured["lora_config"] = lora_config
                captured["training_config"] = training_config

            def train_from_preprocessed(self, tensor_dir, training_state, resume_from=None):
                """Return no training events after configuration is built."""

                return iter([])

        lightning_module = ModuleType("lightning")
        fabric_module = ModuleType("lightning.fabric")
        fabric_module.Fabric = object
        peft_module = ModuleType("peft")
        peft_module.get_peft_model = object
        peft_module.LoraConfig = object
        trainer_module = ModuleType("acestep.training.trainer")
        trainer_module.LoRATrainer = FakeTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            output_dir = os.path.join(tmpdir, "lora")
            handler = SimpleNamespace(model=object(), quantization=None, device="cpu")
            with patch.dict(
                sys.modules,
                {
                    "lightning": lightning_module,
                    "lightning.fabric": fabric_module,
                    "peft": peft_module,
                    "acestep.training.trainer": trainer_module,
                },
            ), patch(
                "acestep.ui.gradio.events.training.lora_training.ensure_dit_ready",
                return_value=(True, ""),
            ), patch(
                "acestep.ui.gradio.events.training.lora_training._training_loss_figure",
                return_value=None,
            ):
                outputs = list(
                    start_training(
                        tmpdir,
                        handler,
                        20,
                        44,
                        0.1,
                        0.0003,
                        10,
                        1,
                        1,
                        10,
                        3.0,
                        42,
                        output_dir,
                        "",
                        {},
                        lora_name="test-lora",
                        gradient_checkpointing=False,
                        activation_cpu_offload=True,
                        offload_non_decoder=False,
                        keep_frozen_base_in_compute_dtype=False,
                        use_8bit_adam=True,
                        base_quantization="FP8 scaled",
                        empty_cache_every_n_steps=17,
                        vram_preset=LORA_VRAM_PRESET_24GB_PLUS,
                    )
                )

        lora_config = captured["lora_config"]
        training_config = captured["training_config"]
        self.assertIn("Training completed", outputs[-1][0])
        self.assertEqual(20, lora_config.r)
        self.assertEqual(44, lora_config.alpha)
        self.assertFalse(training_config.gradient_checkpointing)
        self.assertTrue(training_config.activation_cpu_offload)
        self.assertFalse(training_config.offload_non_decoder)
        self.assertFalse(training_config.keep_frozen_base_in_compute_dtype)
        self.assertTrue(training_config.use_8bit_adam)
        self.assertTrue(training_config.use_fp8)
        self.assertEqual(17, training_config.empty_cache_every_n_steps)

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
