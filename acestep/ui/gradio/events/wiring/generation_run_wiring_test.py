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
        """Inline preview outputs should mirror extracted, remaining, then status."""

        generation_section = {
            "inline_generated_audio": "audio_component",
            "inline_remaining_audio": "remaining_component",
            "inline_generation_status": "status_component",
        }

        self.assertEqual(
            build_inline_result_outputs(generation_section),
            ["audio_component", "remaining_component", "status_component"],
        )

    def test_clear_inline_result_preview_clears_audio_and_status(self):
        """Starting a new generation should clear the inline preview."""

        audio, remaining, status = clear_inline_result_preview()

        self.assertEqual(audio["value"], None)
        self.assertEqual(remaining["value"], None)
        self.assertFalse(remaining["visible"])
        self.assertEqual(status, "")

    def test_prepare_inline_result_preview_shows_extract_progress(self):
        """Starting Extract should show progress in the inline latest-result preview."""

        audio, remaining, status = prepare_inline_result_preview("extract")

        self.assertEqual(audio["value"], None)
        self.assertEqual(audio["label"], "Extracted Audio")
        self.assertEqual(remaining["value"], None)
        self.assertTrue(remaining["visible"])
        self.assertEqual(status, t("messages.extract_stem_processing"))

    def test_prepare_inline_result_preview_keeps_non_extract_behavior(self):
        """Starting non-Extract generation should keep the existing blank status."""

        audio, remaining, status = prepare_inline_result_preview("text2music")

        self.assertEqual(audio["value"], None)
        self.assertEqual(remaining["value"], None)
        self.assertFalse(remaining["visible"])
        self.assertEqual(status, "")

    def test_inline_result_preview_from_generation_outputs_copies_status(self):
        """Streamed generation status should be mirrored into the inline preview."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_remaining.mp3"]

        audio, remaining, status = inline_result_preview_from_generation_outputs(outputs)

        self.assertEqual(audio, "out_0")
        self.assertEqual(remaining["value"], "/tmp/song_remaining.mp3")
        self.assertEqual(remaining["label"], "Original Input")
        self.assertTrue(remaining["visible"])
        self.assertEqual(status, "out_18")

    def test_inline_result_preview_prefers_original_source_audio(self):
        """Source-edit results should show original input beside Sample 1."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_remaining.mp3", "/tmp/source_audio.wav"]

        _audio, original, _status = inline_result_preview_from_generation_outputs(outputs)

        self.assertEqual(original["value"], "/tmp/source_audio.wav")
        self.assertEqual(original["label"], "Original Input")
        self.assertTrue(original["visible"])

    def test_inline_result_preview_from_generation_outputs_preserves_skip_status(self):
        """No-op backend status updates should not erase the inline status."""

        outputs = tuple(["sample.wav", *["unused"] * 15, [], "unused", gr.skip()])

        audio, remaining, status = inline_result_preview_from_generation_outputs(outputs)

        self.assertEqual(audio, "sample.wav")
        self.assertFalse(remaining["visible"])
        self.assertEqual(status, gr.skip())

    def test_append_inline_result_preview_extends_generation_outputs(self):
        """Generation wrapper outputs should include inline audio and status at the end."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_remaining.flac"]
        outputs = tuple(outputs)

        result = append_inline_result_preview(outputs)

        self.assertEqual(result[:-3], outputs)
        self.assertEqual(result[-3], "out_0")
        self.assertEqual(result[-2]["value"], "/tmp/song_remaining.flac")
        self.assertEqual(result[-2]["label"], "Original Input")
        self.assertEqual(result[-1], "out_18")

    def test_sync_inline_result_preview_mirrors_first_sample_and_status(self):
        """Completed generation should copy Sample 1 and status into the preview."""

        audio, remaining, status = sync_inline_result_preview(
            "sample_1.wav",
            ["/tmp/sample_1_remaining.wav"],
            "Generation Complete",
        )

        self.assertEqual(audio, "sample_1.wav")
        self.assertEqual(remaining["value"], "/tmp/sample_1_remaining.wav")
        self.assertEqual(remaining["label"], "Original Input")
        self.assertTrue(remaining["visible"])
        self.assertEqual(status, "Generation Complete")


if __name__ == "__main__":
    unittest.main()
