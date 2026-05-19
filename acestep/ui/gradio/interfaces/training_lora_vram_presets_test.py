"""Tests for LoRA VRAM preset UI helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gradio as gr

from acestep.training.lora_vram_presets import (
    LORA_VRAM_PRESET_10GB,
    LORA_VRAM_PRESET_12GB,
    LORA_VRAM_PRESET_16GB,
    LORA_VRAM_PRESET_24GB,
)
from acestep.ui.gradio.interfaces.training_lora_tab_dataset import (
    build_lora_dataset_and_adapter_controls,
)
from acestep.ui.gradio.interfaces.training_lora_tab_vram import build_lora_vram_controls
from acestep.ui.gradio.interfaces.training_lora_vram_presets import (
    build_lora_vram_preset_dropdown,
    default_lora_vram_control_values,
    lora_vram_preset_updates,
)


class TrainingLoraVramPresetUiTests(unittest.TestCase):
    """Verify the preset dropdown and update contract."""

    def test_dropdown_defaults_to_gpu_selected_preset(self) -> None:
        """The LoRA tab should select the runtime GPU-class preset."""

        with patch(
            "acestep.ui.gradio.interfaces.training_lora_vram_presets."
            "default_lora_vram_preset_name",
            return_value=LORA_VRAM_PRESET_16GB,
        ):
            with gr.Blocks():
                dropdown = build_lora_vram_preset_dropdown()

        self.assertEqual("VRAM Preset", dropdown.label)
        self.assertEqual(LORA_VRAM_PRESET_16GB, dropdown.value)
        choice_values = [choice[1] for choice in dropdown.choices]
        self.assertIn("10 GB - experimental", choice_values)
        self.assertIn("24 GB+ - faster", choice_values)

    def test_default_values_match_selected_dropdown_preset(self) -> None:
        """The startup control values should match the selected default preset."""

        with patch(
            "acestep.ui.gradio.interfaces.training_lora_vram_presets."
            "default_lora_vram_preset_name",
            return_value=LORA_VRAM_PRESET_16GB,
        ):
            defaults = default_lora_vram_control_values()

        self.assertEqual(64, defaults["lora_rank"])
        self.assertEqual(128, defaults["lora_alpha"])
        self.assertTrue(defaults["use_8bit_adam"])
        self.assertEqual(10, defaults["empty_cache_every_n_steps"])

    def test_lora_tabs_start_with_default_preset_values(self) -> None:
        """The initial LoRA tab state should not wait for a dropdown change event."""

        with patch(
            "acestep.ui.gradio.interfaces.training_lora_vram_presets."
            "default_lora_vram_preset_name",
            return_value=LORA_VRAM_PRESET_16GB,
        ):
            with gr.Blocks():
                dataset_controls = build_lora_dataset_and_adapter_controls()
                vram_controls = build_lora_vram_controls()

        self.assertEqual(LORA_VRAM_PRESET_16GB, dataset_controls["lora_vram_preset"].value)
        self.assertEqual(64, dataset_controls["lora_rank"].value)
        self.assertEqual(128, dataset_controls["lora_alpha"].value)
        self.assertEqual("Do not change if you are not expert.", dataset_controls["lora_alpha"].info)
        self.assertEqual(0.0, dataset_controls["lora_dropout"].value)
        self.assertIn("overfitting", dataset_controls["lora_dropout"].info)
        self.assertTrue(vram_controls["lora_use_8bit_adam"].value)
        self.assertEqual(10, vram_controls["lora_empty_cache_every_n_steps"].value)

    def test_preset_updates_match_output_count(self) -> None:
        """Preset changes should update all controlled VRAM widgets."""

        updates = lora_vram_preset_updates(LORA_VRAM_PRESET_12GB)

        self.assertEqual(9, len(updates))
        self.assertEqual(32, updates[0]["value"])
        self.assertTrue(updates[3]["value"])

    def test_preset_updates_match_every_measured_value(self) -> None:
        """Each measured preset should update every dependent LoRA control."""

        expected_values = {
            LORA_VRAM_PRESET_10GB: [16, 128, True, True, True, True, True, "FP8 scaled", 5],
            LORA_VRAM_PRESET_12GB: [32, 128, True, True, True, True, True, "FP8 scaled", 5],
            LORA_VRAM_PRESET_16GB: [64, 128, True, False, True, True, True, "Disabled", 10],
            LORA_VRAM_PRESET_24GB: [128, 128, True, False, True, True, False, "Disabled", 0],
        }

        for preset_name, expected in expected_values.items():
            with self.subTest(preset_name=preset_name):
                updates = lora_vram_preset_updates(preset_name)
                actual = [update["value"] for update in updates]
                self.assertEqual(expected, actual)

    def test_experimental_preset_updates_to_rank_16(self) -> None:
        """The experimental preset should drive the UI to the measured 10 GB values."""

        updates = lora_vram_preset_updates(LORA_VRAM_PRESET_10GB)

        self.assertEqual(9, len(updates))
        self.assertEqual(16, updates[0]["value"])
        self.assertEqual("FP8 scaled", updates[7]["value"])


if __name__ == "__main__":
    unittest.main()
