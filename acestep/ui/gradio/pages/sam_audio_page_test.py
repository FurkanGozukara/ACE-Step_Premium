"""Tests for SAM Audio Segment page construction."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
