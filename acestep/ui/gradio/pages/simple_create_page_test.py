"""Tests for simple Create page startup defaults."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from acestep.ui.gradio.pages.simple_create_page import create_simple_create_page


class SimpleCreatePageTests(unittest.TestCase):
    """Verify runtime-aware defaults for the simple Create tab."""

    def test_uses_global_gpu_config_when_init_params_omit_gpu_config(self) -> None:
        """The simple GPU preset should not start blank without init params."""

        gpu_config = SimpleNamespace(
            tier="tier6a",
            max_duration_without_lm=480,
            quantization_default=True,
        )
        with patch(
            "acestep.ui.gradio.pages.simple_create_page.get_global_gpu_config",
            return_value=gpu_config,
        ):
            with gr.Blocks():
                controls = create_simple_create_page(init_params={"service_mode": False})

        self.assertEqual("tier6a", controls["simple_tier_dropdown"].value)
        self.assertEqual(480, controls["simple_duration"].maximum)
        self.assertEqual("int8_weight_only", controls["simple_quantization"].value)
        self.assertEqual("", controls["simple_negative_prompt"].value)
        self.assertIn("SFT and Base", controls["simple_negative_prompt"].info)

    def test_prefills_simple_lora_controls_from_generation_params(self) -> None:
        """The simple tab should display the same LoRA selection as Advanced."""

        choices = [("None", ""), ("voice", "C:\\Loras\\voice")]
        with patch(
            "acestep.ui.gradio.pages.simple_create_lora_controls.lora_dropdown_choices",
            return_value=choices,
        ):
            with gr.Blocks():
                controls = create_simple_create_page(
                    init_params={
                        "lora_dropdown": "C:\\Loras\\voice",
                        "lora_scale_slider": 2.5,
                    }
                )

        self.assertEqual("C:\\Loras\\voice", controls["simple_lora_dropdown"].value)
        self.assertEqual("Select LoRA", controls["simple_lora_dropdown"].label)
        self.assertEqual("Refresh", controls["simple_refresh_lora_dropdown_btn"].value)
        self.assertEqual(2.5, controls["simple_lora_scale_slider"].value)
        self.assertEqual(3.0, controls["simple_lora_scale_slider"].maximum)


if __name__ == "__main__":
    unittest.main()
