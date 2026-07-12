"""Tests for dataset auto-label UI controls."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.constants import VALID_LANGUAGES
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

    def test_sample_editor_exposes_every_inference_language(self) -> None:
        """LoRA dataset samples should use the same language list as inference."""

        with gr.Blocks():
            controls = build_dataset_label_and_preview_controls()

        choices = controls["edit_language"].choices
        choice_values = {
            choice[1] if isinstance(choice, (list, tuple)) else choice
            for choice in choices
        }
        self.assertEqual({"instrumental", *VALID_LANGUAGES}, choice_values)
        self.assertIn(("Dutch", "nl"), choices)


if __name__ == "__main__":
    unittest.main()
