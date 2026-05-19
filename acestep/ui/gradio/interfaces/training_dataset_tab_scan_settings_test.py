"""Tests for dataset scan/settings UI controls."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gradio as gr

from acestep.training.dataset_vram_presets import DATASET_VRAM_PRESET_CHOICES
from acestep.ui.gradio.interfaces.training_dataset_tab_scan_settings import (
    build_dataset_scan_and_settings_controls,
)


class TrainingDatasetScanSettingsTests(unittest.TestCase):
    """Verify dataset-builder lyric LM options are exposed as usable controls."""

    def test_lm_lyric_options_are_interactive(self) -> None:
        """Format/transcribe checkboxes should be clickable from the UI."""

        with gr.Blocks():
            controls = build_dataset_scan_and_settings_controls()

        self.assertTrue(controls["format_lyrics"].interactive)
        self.assertTrue(controls["transcribe_lyrics"].interactive)
        self.assertFalse(controls["all_instrumental"].value)
        self.assertFalse(controls["format_lyrics"].value)
        self.assertTrue(controls["transcribe_lyrics"].value)
        self.assertEqual("en", controls["lm_lyrics_language"].value)
        self.assertIn(("English", "en"), controls["lm_lyrics_language"].choices)
        self.assertEqual("prepend", controls["tag_position"].value)
        self.assertIn("Genre text instead of the full Caption", controls["genre_ratio"].info)
        self.assertIn("Custom Trigger Tag is filled", controls["tag_position"].info)
        self.assertNotIn("coming soon", controls["format_lyrics"].info.lower())
        self.assertNotIn("coming soon", controls["transcribe_lyrics"].info.lower())

    def test_transcribe_lyrics_defaults_on_for_every_dataset_vram_preset(self) -> None:
        """Every dataset VRAM preset should start with LM transcription enabled."""

        for preset_name in DATASET_VRAM_PRESET_CHOICES:
            with self.subTest(preset_name=preset_name):
                with patch(
                    "acestep.ui.gradio.interfaces.training_dataset_vram_presets."
                    "default_dataset_vram_preset_name",
                    return_value=preset_name,
                ):
                    with gr.Blocks():
                        controls = build_dataset_scan_and_settings_controls()

                self.assertTrue(controls["transcribe_lyrics"].value)


if __name__ == "__main__":
    unittest.main()
