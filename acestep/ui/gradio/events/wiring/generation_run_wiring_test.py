"""Tests for inline result preview wiring helpers."""

import unittest

import gradio as gr

from acestep.ui.gradio.events.wiring.inline_result_preview import (
    append_inline_result_preview,
    build_inline_result_outputs,
    clear_inline_result_preview,
    inline_result_preview_from_generation_outputs,
    prepare_inline_result_preview,
    sync_inline_result_preview,
)
from acestep.ui.gradio.i18n import t


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

    def test_prepare_inline_result_preview_shows_extract_progress(self):
        """Starting Extract should show progress in the inline latest-result preview."""

        self.assertEqual(
            prepare_inline_result_preview("extract"),
            (None, t("messages.extract_stem_processing")),
        )

    def test_prepare_inline_result_preview_keeps_non_extract_behavior(self):
        """Starting non-Extract generation should keep the existing blank status."""

        self.assertEqual(prepare_inline_result_preview("text2music"), (None, ""))

    def test_inline_result_preview_from_generation_outputs_copies_status(self):
        """Streamed generation status should be mirrored into the inline preview."""

        outputs = tuple(f"out_{index}" for index in range(55))

        self.assertEqual(
            inline_result_preview_from_generation_outputs(outputs),
            ("out_0", "out_10"),
        )

    def test_inline_result_preview_from_generation_outputs_preserves_skip_status(self):
        """No-op backend status updates should not erase the inline status."""

        outputs = tuple(["sample.wav", *["unused"] * 9, gr.skip()])

        self.assertEqual(
            inline_result_preview_from_generation_outputs(outputs),
            ("sample.wav", gr.skip()),
        )

    def test_append_inline_result_preview_extends_generation_outputs(self):
        """Generation wrapper outputs should include inline audio and status at the end."""

        outputs = tuple(f"out_{index}" for index in range(55))

        self.assertEqual(
            append_inline_result_preview(outputs),
            (*outputs, "out_0", "out_10"),
        )

    def test_sync_inline_result_preview_mirrors_first_sample_and_status(self):
        """Completed generation should copy Sample 1 and status into the preview."""

        self.assertEqual(
            sync_inline_result_preview("sample_1.wav", "Generation Complete"),
            ("sample_1.wav", "Generation Complete"),
        )


if __name__ == "__main__":
    unittest.main()
