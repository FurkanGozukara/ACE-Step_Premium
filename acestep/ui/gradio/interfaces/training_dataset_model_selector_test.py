"""Tests for the training dataset model selector."""

import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_dataset_model_selector import (
    build_dataset_model_selector,
)


class _FakeDitHandler:
    """Test double returning available DiT models."""

    def __init__(self):
        self.last_init_params = {"config_path": "model-b"}

    def get_available_acestep_v15_models(self):
        """Return selectable models for the dropdown."""

        return ["model-a", "model-b"]


class TrainingDatasetModelSelectorTests(unittest.TestCase):
    """Tests for dataset model dropdown defaults."""

    def test_uses_init_params_before_last_initialized_model(self):
        """Startup params should seed the dataset model dropdown."""

        with gr.Blocks():
            controls = build_dataset_model_selector(
                _FakeDitHandler(),
                init_params={"config_path": "model-a"},
            )

        self.assertEqual(controls["dataset_model_config"].value, "model-a")

    def test_falls_back_to_last_initialized_model(self):
        """Last initialized DiT model should be used when startup params are absent."""

        with gr.Blocks():
            controls = build_dataset_model_selector(
                _FakeDitHandler(),
                init_params=None,
            )

        self.assertEqual(controls["dataset_model_config"].value, "model-b")

    def test_handles_missing_handler(self):
        """Missing handlers should still produce a usable custom-value dropdown."""

        with gr.Blocks():
            controls = build_dataset_model_selector(None)

        dropdown = controls["dataset_model_config"]
        self.assertEqual(dropdown.value, "ACEStep_1_5_XL_Turbo_BF16")
        self.assertEqual(dropdown.choices, [])


if __name__ == "__main__":
    unittest.main()
