"""Tests for Audio Processing preset UI actions."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.audio_processing.presets import (
    PROCESSING_PRESET_NONE,
    PRESET_VALUES,
    STAGE_KEYS,
)
from acestep.ui.gradio.events.wiring.audio_processing_preset_actions import (
    apply_builtin_preset,
)
from acestep.ui.gradio.pages.audio_processing_page import create_audio_processing_page


class AudioProcessingPresetActionTests(unittest.TestCase):
    """Verify Processing Preset stage-enable behavior."""

    def test_none_preset_is_available_in_processing_preset_dropdown(self) -> None:
        """The Processing Preset dropdown should expose the None preset."""

        with gr.Blocks():
            controls = create_audio_processing_page()

        choice_values = [choice[1] for choice in controls["ap_builtin_preset"].choices]

        self.assertIn(PROCESSING_PRESET_NONE, choice_values)

    def test_none_processing_preset_unchecks_all_stages(self) -> None:
        """The None preset should disable every Audio Processing stage."""

        updates = apply_builtin_preset(PROCESSING_PRESET_NONE)
        stage_updates = updates[: len(STAGE_KEYS)]
        enabled_updates = updates[len(STAGE_KEYS):]

        self.assertEqual(len(STAGE_KEYS) * 2, len(updates))
        self.assertEqual(
            PRESET_VALUES[PROCESSING_PRESET_NONE]["lufs"],
            stage_updates[-1]["value"],
        )
        self.assertTrue(all(update["value"] is False for update in enabled_updates))

    def test_regular_processing_preset_checks_all_stages(self) -> None:
        """Selecting a normal preset after None should re-enable all stages."""

        updates = apply_builtin_preset("Generic AI")
        enabled_updates = updates[len(STAGE_KEYS):]

        self.assertTrue(all(update["value"] is True for update in enabled_updates))


if __name__ == "__main__":
    unittest.main()
