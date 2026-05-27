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
        self.assertEqual(0, controls["save_every_n_epochs"].minimum)
        self.assertEqual(default_lora_output_dir(), controls["lora_output_dir"].value)
        self.assertTrue(controls["lora_save_best"].value)
        self.assertEqual(10, controls["lora_save_best_after"].value)
        self.assertEqual(5, controls["lora_save_best_smoothing_window"].value)
        self.assertEqual(0.001, controls["lora_save_best_min_delta"].value)
        self.assertEqual("constant", controls["lora_scheduler_type"].value)
        self.assertEqual(0, controls["lora_validation_split_percent"].value)
        self.assertIn("validation split", controls["lora_validation_split_info"].value.lower())
        self.assertTrue(os.path.isabs(controls["lora_output_dir"].value))
        self.assertTrue(controls["lora_output_dir"].value.endswith(os.path.join("", "Loras")))
        self.assertEqual("Open Output Folder", controls["lora_open_output_dir_btn"].value)
        self.assertIn("training_step_estimate", controls)
        self.assertEqual(8, controls["training_num_inference_steps"].value)
        self.assertEqual(200, controls["training_num_inference_steps"].maximum)
        self.assertEqual("Resume Training State", controls["resume_checkpoint_dir"].label)
        self.assertEqual(
            "Browse Training State",
            controls["resume_checkpoint_dir_browse_btn"].value,
        )
        self.assertNotIn("export_lora_btn", controls)
        self.assertNotIn("export_path", controls)

    def test_sft_model_sets_schedule_defaults(self) -> None:
        """Initial LoRA schedule controls should follow the selected base model."""

        with gr.Blocks():
            controls = build_lora_run_and_export_controls(
                epoch_min=1,
                epoch_step=1,
                epoch_default=100,
                model_config="acestep-v15-xl-sft",
            )

        self.assertEqual(1.0, controls["training_shift"].value)
        self.assertEqual(50, controls["training_num_inference_steps"].value)
        self.assertEqual(200, controls["training_num_inference_steps"].maximum)


if __name__ == "__main__":
    unittest.main()
