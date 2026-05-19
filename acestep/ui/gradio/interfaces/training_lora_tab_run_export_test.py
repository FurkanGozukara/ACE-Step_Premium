"""Tests for LoRA run/export UI controls."""

from __future__ import annotations

import os
import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_lora_tab_run_export import (
    build_lora_run_and_export_controls,
    default_lora_output_dir,
)


class TrainingLoraRunExportControlsTests(unittest.TestCase):
    """Verify LoRA run/export control defaults."""

    def test_epochs_and_output_defaults(self) -> None:
        """The LoRA run panel should use requested training defaults."""

        with gr.Blocks():
            controls = build_lora_run_and_export_controls(
                epoch_min=1,
                epoch_step=1,
                epoch_default=100,
            )

        self.assertEqual("Training Epochs Count", controls["train_epochs"].label)
        self.assertEqual(100, controls["train_epochs"].value)
        self.assertEqual(default_lora_output_dir(), controls["lora_output_dir"].value)
        self.assertTrue(os.path.isabs(controls["lora_output_dir"].value))
        self.assertTrue(controls["lora_output_dir"].value.endswith(os.path.join("", "Loras")))
        self.assertIn("training_step_estimate", controls)


if __name__ == "__main__":
    unittest.main()
