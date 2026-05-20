"""Tests for dataset VRAM preset UI helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gradio as gr

from acestep.training.dataset_vram_presets import DATASET_VRAM_PRESET_16GB
from acestep.ui.gradio.interfaces.training_dataset_vram_presets import (
    build_dataset_vram_preset_dropdown,
)


class TrainingDatasetVramPresetUiTests(unittest.TestCase):
    """Verify the dataset VRAM preset dropdown contract."""

    def test_dropdown_exposes_measured_presets(self) -> None:
        """Dataset tab should expose safe and quality preset choices."""

        with patch(
            "acestep.ui.gradio.interfaces.training_dataset_vram_presets."
            "default_dataset_vram_preset_name",
            return_value=DATASET_VRAM_PRESET_16GB,
        ):
            with gr.Blocks():
                dropdown = build_dataset_vram_preset_dropdown()

        values = [choice[1] for choice in dropdown.choices]
        self.assertEqual("Dataset VRAM Preset", dropdown.label)
        self.assertEqual(DATASET_VRAM_PRESET_16GB, dropdown.value)
        self.assertIn("10 GB+", values)
        self.assertIn("12-16 GB", values)
        self.assertIn("24 GB+ - quality", values)


if __name__ == "__main__":
    unittest.main()
