"""Tests for LoRA optimizer option controls."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_lora_tab_training_options import (
    LORA_OPTIMIZER_HYPERPARAMETER_KEYS,
    LORA_OPTIMIZER_PARAMETER_ROW_KEYS,
    build_lora_training_option_controls,
    lora_optimizer_hyperparameter_updates,
    lora_optimizer_parameter_row_updates,
)


def _updates_by_key(optimizer_type: str) -> dict[str, object]:
    return dict(
        zip(
            LORA_OPTIMIZER_HYPERPARAMETER_KEYS,
            lora_optimizer_hyperparameter_updates(optimizer_type),
        )
    )


def _row_updates_by_key(optimizer_type: str) -> dict[str, object]:
    return dict(
        zip(
            LORA_OPTIMIZER_PARAMETER_ROW_KEYS,
            lora_optimizer_parameter_row_updates(optimizer_type),
        )
    )


class TrainingLoraTabTrainingOptionsTests(unittest.TestCase):
    """Verify optimizer-specific hidden controls and defaults."""

    def test_optimizer_parameter_controls_render_below_optimizer_row(self) -> None:
        """Optimizer params should be direct controls, not buried in an accordion."""

        with gr.Blocks() as demo:
            build_lora_training_option_controls("adamw")

        labels = [
            component.get("props", {}).get("label")
            for component in demo.config["components"]
        ]
        self.assertIn("Optimizer", labels)
        self.assertIn("Weight decay", labels)
        self.assertNotIn("Optimizer parameters", labels)
        self.assertGreater(labels.index("Weight decay"), labels.index("Optimizer"))

    def test_adamw_defaults_show_adam_fields_only(self) -> None:
        """AdamW should show Adam hyperparameters and hide unrelated fields."""

        updates = _updates_by_key("adamw")

        self.assertEqual(0.01, updates["lora_weight_decay"].get("value"))
        self.assertTrue(updates["lora_weight_decay"].get("visible"))
        self.assertTrue(updates["lora_adam_beta1"].get("visible"))
        self.assertTrue(updates["lora_adam_beta2"].get("visible"))
        self.assertTrue(updates["lora_adam_epsilon"].get("visible"))
        self.assertTrue(updates["lora_adamw8bit_min_8bit_size"].get("visible"))
        self.assertEqual(1e-30, updates["lora_adafactor_epsilon1"].get("value"))
        self.assertEqual(1e-3, updates["lora_adafactor_epsilon2"].get("value"))
        self.assertFalse(updates["lora_adafactor_epsilon1"].get("visible"))
        self.assertFalse(updates["lora_adafactor_scale_parameter"].get("value"))
        self.assertFalse(updates["lora_adafactor_relative_step"].get("value"))
        self.assertFalse(updates["lora_adafactor_warmup_init"].get("value"))
        rows = _row_updates_by_key("adamw")
        self.assertTrue(rows["lora_adam_parameters_row"].get("visible"))
        self.assertFalse(rows["lora_adamw8bit_parameters_row"].get("visible"))
        self.assertFalse(rows["lora_adafactor_parameters_row"].get("visible"))

    def test_adamw8bit_defaults_show_bitsandbytes_fields(self) -> None:
        """AdamW8bit should show Adam and bitsandbytes-specific parameters."""

        updates = _updates_by_key("adamw8bit")

        self.assertEqual(0.01, updates["lora_weight_decay"].get("value"))
        self.assertEqual(0.9, updates["lora_adam_beta1"].get("value"))
        self.assertEqual(0.999, updates["lora_adam_beta2"].get("value"))
        self.assertEqual(1e-8, updates["lora_adam_epsilon"].get("value"))
        self.assertEqual(4096, updates["lora_adamw8bit_min_8bit_size"].get("value"))
        self.assertEqual(
            100,
            updates["lora_adamw8bit_percentile_clipping"].get("value"),
        )
        self.assertTrue(updates["lora_adamw8bit_block_wise"].get("value"))
        self.assertFalse(updates["lora_adamw8bit_paged"].get("value"))
        self.assertTrue(updates["lora_adamw8bit_min_8bit_size"].get("visible"))
        self.assertEqual(1e-30, updates["lora_adafactor_epsilon1"].get("value"))
        self.assertFalse(updates["lora_adafactor_epsilon1"].get("visible"))
        self.assertFalse(updates["lora_adafactor_scale_parameter"].get("value"))
        self.assertFalse(updates["lora_adafactor_relative_step"].get("value"))
        self.assertFalse(updates["lora_adafactor_warmup_init"].get("value"))
        rows = _row_updates_by_key("adamw8bit")
        self.assertTrue(rows["lora_adam_parameters_row"].get("visible"))
        self.assertTrue(rows["lora_adamw8bit_parameters_row"].get("visible"))
        self.assertFalse(rows["lora_adafactor_parameters_row"].get("visible"))

    def test_adafactor_defaults_show_manual_lr_safe_fields(self) -> None:
        """Adafactor should default to external LR/scheduler-compatible settings."""

        updates = _updates_by_key("adafactor")

        self.assertEqual(0.0, updates["lora_weight_decay"].get("value"))
        self.assertTrue(updates["lora_adam_beta1"].get("visible"))
        self.assertTrue(updates["lora_adamw8bit_min_8bit_size"].get("visible"))
        self.assertEqual(1e-30, updates["lora_adafactor_epsilon1"].get("value"))
        self.assertEqual(1e-3, updates["lora_adafactor_epsilon2"].get("value"))
        self.assertEqual(1.0, updates["lora_adafactor_clip_threshold"].get("value"))
        self.assertEqual(-0.8, updates["lora_adafactor_decay_rate"].get("value"))
        self.assertEqual(0.0, updates["lora_adafactor_beta1"].get("value"))
        self.assertTrue(updates["lora_adafactor_epsilon1"].get("visible"))
        self.assertTrue(updates["lora_adafactor_beta1"].get("visible"))
        self.assertFalse(updates["lora_adafactor_scale_parameter"].get("value"))
        self.assertFalse(updates["lora_adafactor_relative_step"].get("value"))
        self.assertFalse(updates["lora_adafactor_warmup_init"].get("value"))
        rows = _row_updates_by_key("adafactor")
        self.assertFalse(rows["lora_adam_parameters_row"].get("visible"))
        self.assertFalse(rows["lora_adamw8bit_parameters_row"].get("visible"))
        self.assertTrue(rows["lora_adafactor_parameters_row"].get("visible"))


if __name__ == "__main__":
    unittest.main()
