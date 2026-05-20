"""Tests for LoRA training VRAM presets."""

from __future__ import annotations

import unittest

from acestep.training.lora_vram_presets import (
    LORA_VRAM_PRESET_8_TO_10GB,
    LORA_VRAM_PRESET_10GB_PLUS,
    LORA_VRAM_PRESET_12_TO_16GB,
    LORA_VRAM_PRESET_16_TO_24GB,
    LORA_VRAM_PRESET_24GB_PLUS,
    LORA_VRAM_PRESET_MANUAL,
    apply_lora_vram_preset,
    get_lora_vram_preset,
    select_lora_vram_preset_for_gpu,
)


class LoraVramPresetTests(unittest.TestCase):
    """Verify preset lookup and runtime value behavior."""

    def test_apply_helper_keeps_submitted_values_for_any_preset(self) -> None:
        """Preset application should preserve submitted Gradio values."""

        values = {"lora_rank": 64, "use_8bit_adam": False}

        for preset_name in (LORA_VRAM_PRESET_MANUAL, LORA_VRAM_PRESET_24GB_PLUS):
            with self.subTest(preset_name=preset_name):
                self.assertEqual(values, apply_lora_vram_preset(preset_name, values))

    def test_10gb_plus_preset_enables_memory_savers(self) -> None:
        """The 10 GB+ preset should enable aggressive VRAM-saving settings."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_10GB_PLUS)

        self.assertEqual(32, preset["lora_rank"])
        self.assertTrue(preset["gradient_checkpointing"])
        self.assertTrue(preset["activation_cpu_offload"])
        self.assertTrue(preset["use_8bit_adam"])
        self.assertEqual("FP8 scaled", preset["base_quantization"])

    def test_8_to_10gb_preset_uses_lowest_measured_rank(self) -> None:
        """The 8-10 GB preset should use the lowest measured working LoRA rank."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_8_TO_10GB)

        self.assertEqual(16, preset["lora_rank"])
        self.assertEqual(128, preset["lora_alpha"])
        self.assertTrue(preset["activation_cpu_offload"])
        self.assertEqual("FP8 scaled", preset["base_quantization"])

    def test_all_presets_keep_alpha_128(self) -> None:
        """Every automatic preset should keep the expert-only alpha default."""

        for preset_name in (
            LORA_VRAM_PRESET_8_TO_10GB,
            LORA_VRAM_PRESET_10GB_PLUS,
            LORA_VRAM_PRESET_12_TO_16GB,
            LORA_VRAM_PRESET_16_TO_24GB,
            LORA_VRAM_PRESET_24GB_PLUS,
        ):
            with self.subTest(preset_name=preset_name):
                self.assertEqual(128, get_lora_vram_preset(preset_name)["lora_alpha"])

    def test_16_to_24gb_preset_raises_rank_without_cpu_offload(self) -> None:
        """The 16-24 GB preset should prefer quality and speed."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_16_TO_24GB)

        self.assertEqual(128, preset["lora_rank"])
        self.assertFalse(preset["activation_cpu_offload"])
        self.assertTrue(preset["keep_frozen_base_in_compute_dtype"])
        self.assertFalse(preset["use_8bit_adam"])

    def test_24gb_plus_preset_disables_frozen_compute_dtype_saver(self) -> None:
        """The 24GB+ preset should uncheck frozen-base bf16/fp16."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_24GB_PLUS)

        self.assertEqual(128, preset["lora_rank"])
        self.assertFalse(preset["activation_cpu_offload"])
        self.assertFalse(preset["keep_frozen_base_in_compute_dtype"])
        self.assertFalse(preset["use_8bit_adam"])

    def test_default_preset_selection_uses_gpu_vram_class(self) -> None:
        """Runtime defaults should choose the measured preset for the GPU class."""

        self.assertEqual(
            LORA_VRAM_PRESET_24GB_PLUS,
            select_lora_vram_preset_for_gpu(31.8),
        )
        self.assertEqual(
            LORA_VRAM_PRESET_24GB_PLUS,
            select_lora_vram_preset_for_gpu(23.31),
        )
        self.assertEqual(
            LORA_VRAM_PRESET_16_TO_24GB,
            select_lora_vram_preset_for_gpu(23.3),
        )
        self.assertEqual(
            LORA_VRAM_PRESET_16_TO_24GB,
            select_lora_vram_preset_for_gpu(15.5),
        )
        self.assertEqual(
            LORA_VRAM_PRESET_10GB_PLUS,
            select_lora_vram_preset_for_gpu(15.49),
        )
        self.assertEqual(
            LORA_VRAM_PRESET_10GB_PLUS,
            select_lora_vram_preset_for_gpu(10.0),
        )
        self.assertEqual(
            LORA_VRAM_PRESET_8_TO_10GB,
            select_lora_vram_preset_for_gpu(9.99),
        )


if __name__ == "__main__":
    unittest.main()
