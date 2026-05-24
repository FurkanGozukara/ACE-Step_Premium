"""Tests for dataset save and tensor-preprocess UI controls."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_dataset_tab_save_preprocess import (
    build_dataset_save_and_preprocess_controls,
)


class TrainingDatasetSavePreprocessTests(unittest.TestCase):
    """Verify tensor-preprocess controls are exposed with safe defaults."""

    def test_preprocess_cancel_control_sits_with_progress(self) -> None:
        """The preprocess progress row should expose a cancel action."""

        with gr.Blocks():
            controls = build_dataset_save_and_preprocess_controls()

        self.assertEqual("Cancel Preprocess", controls["cancel_preprocess_btn"].value)
        self.assertFalse(controls["preprocess_progress"].interactive)
        self.assertTrue(controls["preprocess_subprocess"].value)


if __name__ == "__main__":
    unittest.main()
