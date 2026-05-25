"""Tests for Grid Testing page defaults."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gradio as gr

from acestep.ui.gradio.pages.grid_testing_page import create_grid_testing_page


class GridTestingPageTests(unittest.TestCase):
    """Verify Grid Testing controls start with safe defaults."""

    def test_mp3_only_is_checked_by_default(self) -> None:
        """Grid runs should default to MP3-only output to avoid clutter."""

        with patch(
            "acestep.ui.gradio.pages.grid_testing_page.lora_dropdown_choices",
            return_value=[("None", ""), ("voice", "C:\\Loras\\voice.safetensors")],
        ):
            with gr.Blocks():
                controls = create_grid_testing_page()

        self.assertEqual([""], controls["grid_lora_dropdown"].value)
        self.assertEqual("", controls["grid_lora_filter"].value)
        self.assertTrue(controls["grid_mp3_only"].value)


if __name__ == "__main__":
    unittest.main()
