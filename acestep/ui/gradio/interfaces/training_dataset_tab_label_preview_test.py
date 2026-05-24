"""Tests for dataset auto-label UI controls."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_dataset_tab_label_preview import (
    build_dataset_label_and_preview_controls,
)


class TrainingDatasetLabelPreviewTests(unittest.TestCase):
    """Verify auto-label controls are exposed with safe defaults."""

    def test_auto_label_batch_size_slider_defaults_to_one(self) -> None:
        """The batch-size slider should expose the supported 1-99 range."""

        with gr.Blocks():
            controls = build_dataset_label_and_preview_controls()

        slider = controls["auto_label_batch_size"]
        self.assertEqual(1, slider.value)
        self.assertEqual(1, slider.minimum)
        self.assertEqual(99, slider.maximum)
        self.assertEqual(1, slider.step)
        self.assertTrue(controls["auto_label_subprocess"].value)

    def test_auto_label_cancel_control_sits_with_progress(self) -> None:
        """The progress row should expose a confirmed cancel action."""

        with gr.Blocks():
            controls = build_dataset_label_and_preview_controls()

        self.assertEqual("Cancel Auto-Label", controls["cancel_auto_label_btn"].value)
        self.assertFalse(controls["label_progress"].interactive)


if __name__ == "__main__":
    unittest.main()
