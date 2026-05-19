"""Tests for LoRA training VRAM presets."""

from __future__ import annotations

import unittest

from acestep.training.lora_vram_presets import (
    LORA_VRAM_PRESET_10GB,
    LORA_VRAM_PRESET_12GB,
    LORA_VRAM_PRESET_16GB,
    LORA_VRAM_PRESET_24GB,
    LORA_VRAM_PRESET_MANUAL,
    apply_lora_vram_preset,
    get_lora_vram_preset,
    select_lora_vram_preset_for_gpu,
)


class LoraVramPresetTests(unittest.TestCase):
    """Verify preset lookup and overlay behavior."""

    def test_manual_preset_keeps_existing_values(self) -> None:
        """Manual mode should not alter user-selected training controls."""

        values = {"lora_rank": 64, "use_8bit_adam": False}

        self.assertEqual(values, apply_lora_vram_preset(LORA_VRAM_PRESET_MANUAL, values))

    def test_safest_preset_enables_memory_savers(self) -> None:
        """The 12 GB preset should enable aggressive VRAM-saving settings."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_12GB)

        self.assertEqual(32, preset["lora_rank"])
        self.assertTrue(preset["gradient_checkpointing"])
        self.assertTrue(preset["activation_cpu_offload"])
        self.assertTrue(preset["use_8bit_adam"])
        self.assertEqual("FP8 scaled", preset["base_quantization"])

    def test_experimental_preset_uses_lowest_measured_rank(self) -> None:
        """The 10 GB preset should use the lowest measured working LoRA rank."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_10GB)

        self.assertEqual(16, preset["lora_rank"])
        self.assertEqual(32, preset["lora_alpha"])
        self.assertTrue(preset["activation_cpu_offload"])
        self.assertEqual("FP8 scaled", preset["base_quantization"])

    def test_faster_preset_raises_rank_without_cpu_activation_offload(self) -> None:
        """The 24 GB preset should prefer speed and higher rank."""

        preset = get_lora_vram_preset(LORA_VRAM_PRESET_24GB)

        self.assertEqual(128, preset["lora_rank"])
        self.assertFalse(preset["activation_cpu_offload"])
        self.assertFalse(preset["use_8bit_adam"])

    def test_default_preset_selection_uses_gpu_vram_class(self) -> None:
        """Runtime defaults should choose the measured preset for the GPU class."""

        self.assertEqual(LORA_VRAM_PRESET_24GB, select_lora_vram_preset_for_gpu(31.8))
        self.assertEqual(LORA_VRAM_PRESET_24GB, select_lora_vram_preset_for_gpu(23.8))
        self.assertEqual(LORA_VRAM_PRESET_16GB, select_lora_vram_preset_for_gpu(15.5))
        self.assertEqual(LORA_VRAM_PRESET_12GB, select_lora_vram_preset_for_gpu(11.8))
        self.assertEqual(LORA_VRAM_PRESET_10GB, select_lora_vram_preset_for_gpu(10.0))


if __name__ == "__main__":
    unittest.main()
