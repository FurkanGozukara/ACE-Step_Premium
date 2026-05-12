"""Tests for inline result preview wiring helpers."""

import unittest

from acestep.ui.gradio.events.wiring.inline_result_preview import (
    build_inline_result_outputs,
    clear_inline_result_preview,
    sync_inline_result_preview,
)


class InlineResultPreviewTests(unittest.TestCase):
    """Verify inline latest-result preview helper behavior."""

    def test_build_inline_result_outputs_uses_expected_order(self):
        """Inline preview outputs should mirror audio first, then status."""

        generation_section = {
            "inline_generated_audio": "audio_component",
            "inline_generation_status": "status_component",
        }

        self.assertEqual(
            build_inline_result_outputs(generation_section),
            ["audio_component", "status_component"],
        )

    def test_clear_inline_result_preview_clears_audio_and_status(self):
        """Starting a new generation should clear the inline preview."""

        self.assertEqual(clear_inline_result_preview(), (None, ""))

    def test_sync_inline_result_preview_mirrors_first_sample_and_status(self):
        """Completed generation should copy Sample 1 and status into the preview."""

        self.assertEqual(
            sync_inline_result_preview("sample_1.wav", "Generation Complete"),
            ("sample_1.wav", "Generation Complete"),
        )


if __name__ == "__main__":
    unittest.main()
