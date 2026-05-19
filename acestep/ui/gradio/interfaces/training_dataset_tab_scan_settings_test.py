"""Tests for dataset scan/settings UI controls."""

from __future__ import annotations

import unittest

import gradio as gr

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
        self.assertTrue(controls["format_lyrics"].value)
        self.assertTrue(controls["transcribe_lyrics"].value)
        self.assertEqual("en", controls["lm_lyrics_language"].value)
        self.assertIn(("English", "en"), controls["lm_lyrics_language"].choices)
        self.assertEqual("prepend", controls["tag_position"].value)
        self.assertIn("Genre text instead of the full Caption", controls["genre_ratio"].info)
        self.assertIn("Custom Trigger Tag is filled", controls["tag_position"].info)
        self.assertNotIn("coming soon", controls["format_lyrics"].info.lower())
        self.assertNotIn("coming soon", controls["transcribe_lyrics"].info.lower())


if __name__ == "__main__":
    unittest.main()
