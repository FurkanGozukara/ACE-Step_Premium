"""Tests for dataset action VRAM presets."""

from __future__ import annotations

import unittest

from acestep.training.dataset_vram_presets import (
    DATASET_VRAM_PRESET_12GB,
    DATASET_VRAM_PRESET_16GB,
    DATASET_VRAM_PRESET_24GB,
    DATASET_VRAM_PRESET_AUTO,
    apply_dataset_dit_preset,
    apply_dataset_llm_preset,
    dataset_vram_preset_requires_subprocess,
    get_dataset_vram_preset,
    select_dataset_vram_preset_for_gpu,
)


class DatasetVramPresetTests(unittest.TestCase):
    """Verify dataset action preset lookup and overlays."""

    def test_auto_preset_keeps_existing_values(self) -> None:
        """Auto mode should not alter init payloads."""

        params = {"quantization": None, "offload_to_cpu": False}

        self.assertEqual(
            params,
            apply_dataset_dit_preset(
                DATASET_VRAM_PRESET_AUTO,
                params,
                operation="preprocess",
            ),
        )
        self.assertFalse(dataset_vram_preset_requires_subprocess(DATASET_VRAM_PRESET_AUTO))

    def test_safest_preset_uses_measured_low_vram_models(self) -> None:
        """The 12 GB preset should use the measured low-VRAM dataset settings."""

        preset = get_dataset_vram_preset(DATASET_VRAM_PRESET_12GB)

        self.assertEqual("acestep-5Hz-lm-0.6B", preset["auto_label"]["llm"]["lm_model_path"])
        self.assertEqual("fp8_scaled", preset["auto_label"]["dit"]["quantization"])
        self.assertEqual("int8_weight_only", preset["preprocess"]["dit"]["quantization"])
        self.assertFalse(preset["preprocess"]["dit"]["offload_dit_to_cpu"])

    def test_balanced_preset_uses_mid_tier_lm(self) -> None:
        """The 16 GB preset should use the 1.7B LM and module offload."""

        llm_params = apply_dataset_llm_preset(
            DATASET_VRAM_PRESET_16GB,
            {"lm_model_path": "old", "backend": "vllm"},
        )

        self.assertEqual("acestep-5Hz-lm-1.7B", llm_params["lm_model_path"])
        self.assertEqual("pt", llm_params["backend"])

    def test_quality_preset_keeps_preprocess_unquantized(self) -> None:
        """The 24 GB preset should prefer unquantized preprocessing."""

        dit_params = apply_dataset_dit_preset(
            DATASET_VRAM_PRESET_24GB,
            {"quantization": "int8_weight_only"},
            operation="preprocess",
        )

        self.assertIsNone(dit_params["quantization"])
        self.assertFalse(dit_params["offload_to_cpu"])

    def test_default_preset_selection_uses_gpu_vram_class(self) -> None:
        """Runtime defaults should choose the measured dataset preset for the GPU class."""

        self.assertEqual(
            DATASET_VRAM_PRESET_24GB,
            select_dataset_vram_preset_for_gpu(31.8),
        )
        self.assertEqual(
            DATASET_VRAM_PRESET_24GB,
            select_dataset_vram_preset_for_gpu(23.8),
        )
        self.assertEqual(
            DATASET_VRAM_PRESET_16GB,
            select_dataset_vram_preset_for_gpu(15.5),
        )
        self.assertEqual(
            DATASET_VRAM_PRESET_12GB,
            select_dataset_vram_preset_for_gpu(11.8),
        )
        self.assertEqual(
            DATASET_VRAM_PRESET_12GB,
            select_dataset_vram_preset_for_gpu(8.0),
        )


if __name__ == "__main__":
    unittest.main()
