"""Tests for LoRA checkpoint sample controls."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_lora_tab_samples import (
    build_lora_sample_generation_controls,
)


class TrainingLoraTabSamplesTests(unittest.TestCase):
    """Verify checkpoint sample controls stay scoped to run-folder outputs."""

    def test_sample_output_directory_control_is_not_rendered(self) -> None:
        """Checkpoint samples should not expose a separate output folder."""

        with gr.Blocks():
            controls = build_lora_sample_generation_controls()

        self.assertIn("lora_sample_enabled", controls)
        self.assertNotIn("lora_sample_output_dir", controls)
        self.assertNotIn("lora_sample_output_dir_browse_btn", controls)


if __name__ == "__main__":
    unittest.main()
