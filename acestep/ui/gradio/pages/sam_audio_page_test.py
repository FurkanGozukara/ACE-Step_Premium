"""Tests for SAM Audio Segment page construction."""

import unittest
from unittest.mock import patch

import gradio as gr

from acestep.sam_audio_segment.settings import SAM_AUDIO_PRESET_KEYS
from acestep.ui.gradio.pages.sam_audio_page import create_sam_audio_page


class SamAudioPageTests(unittest.TestCase):
    """Verify SAM page controls stay aligned with settings serialization."""

    def test_page_exposes_every_preset_key(self) -> None:
        """Every SAM preset key should have a concrete page component."""

        demo = gr.Blocks()
        try:
            with demo:
                controls = create_sam_audio_page()
        finally:
            demo.close()

        missing = [key for key in SAM_AUDIO_PRESET_KEYS if key not in controls]
        self.assertEqual([], missing)

    def test_predict_spans_checkbox_uses_full_model_default_state(self) -> None:
        """Predict-spans should be usable when the default SAM preset loads the module."""

        demo = gr.Blocks()
        try:
            with demo, patch(
                "acestep.ui.gradio.pages.sam_audio_page_settings."
                "default_sam_vram_preset_name",
                return_value="32gb_quality",
            ):
                controls = create_sam_audio_page()
        finally:
            demo.close()

        self.assertFalse(controls["sam_predict_spans"].value)
        self.assertTrue(controls["sam_predict_spans"].interactive)

    def test_quick_prompt_dropdown_is_multiselect(self) -> None:
        """Quick Prompt should accept multiple Batch Segment targets."""

        demo = gr.Blocks()
        try:
            with demo:
                controls = create_sam_audio_page()
        finally:
            demo.close()

        self.assertTrue(controls["sam_prompt_preset"].multiselect)
        self.assertEqual(["vocals"], controls["sam_prompt_preset"].value)


if __name__ == "__main__":
    unittest.main()
