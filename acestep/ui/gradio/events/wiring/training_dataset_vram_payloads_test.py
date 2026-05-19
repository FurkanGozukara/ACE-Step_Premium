"""Tests for dataset VRAM preset payload helpers."""

from __future__ import annotations

import unittest

from acestep.training.dataset_vram_presets import (
    DATASET_VRAM_PRESET_12GB,
    DATASET_VRAM_PRESET_AUTO,
)
from acestep.ui.gradio.events.wiring.training_dataset_vram_payloads import (
    build_auto_label_init_payloads,
    build_preprocess_dit_init_payload,
    should_run_dataset_action_in_subprocess,
)


class _Handler:
    """Small handler double exposing last init params."""

    def __init__(self, params: dict | None = None) -> None:
        self.last_init_params = params or {}


class TrainingDatasetVramPayloadTests(unittest.TestCase):
    """Verify preset payload overlays for dataset workers."""

    def test_auto_preset_respects_subprocess_checkbox(self) -> None:
        """Auto mode should only use subprocess when the checkbox asks for it."""

        self.assertFalse(
            should_run_dataset_action_in_subprocess(DATASET_VRAM_PRESET_AUTO, False)
        )
        self.assertTrue(
            should_run_dataset_action_in_subprocess(DATASET_VRAM_PRESET_AUTO, True)
        )

    def test_low_vram_preset_forces_subprocess(self) -> None:
        """Measured presets should force a worker so init settings can be controlled."""

        self.assertTrue(
            should_run_dataset_action_in_subprocess(DATASET_VRAM_PRESET_12GB, False)
        )

    def test_auto_label_payload_applies_dit_and_lm_settings(self) -> None:
        """Auto-label preset should overlay both DiT and LM init params."""

        dit_params, llm_params = build_auto_label_init_payloads(
            _Handler({"project_root": "root", "quantization": None}),
            _Handler({"lm_model_path": "old", "backend": "vllm"}),
            "model-a",
            DATASET_VRAM_PRESET_12GB,
        )

        self.assertEqual("model-a", dit_params["config_path"])
        self.assertEqual("fp8_scaled", dit_params["quantization"])
        self.assertTrue(dit_params["offload_dit_to_cpu"])
        self.assertEqual("acestep-5Hz-lm-0.6B", llm_params["lm_model_path"])
        self.assertEqual("pt", llm_params["backend"])

    def test_preprocess_payload_keeps_dit_on_gpu_for_low_vram(self) -> None:
        """Preprocess preset should use the measured working INT8/offload profile."""

        dit_params = build_preprocess_dit_init_payload(
            _Handler({"project_root": "root", "quantization": None}),
            "model-a",
            DATASET_VRAM_PRESET_12GB,
        )

        self.assertEqual("int8_weight_only", dit_params["quantization"])
        self.assertTrue(dit_params["offload_to_cpu"])
        self.assertFalse(dit_params["offload_dit_to_cpu"])


if __name__ == "__main__":
    unittest.main()
