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
            "inline_repainted_area_audio": "area_component",
            "inline_repainted_area_original_audio": "area_original_component",
            "inline_lego_part_audio": "lego_part_component",
            "inline_generation_status": "status_component",
        }

        self.assertEqual(
            build_inline_result_outputs(generation_section),
            [
                "audio_component",
                "remaining_component",
                "area_component",
                "area_original_component",
                "lego_part_component",
                "status_component",
            ],
        )

    def test_clear_inline_result_preview_clears_audio_and_status(self):
        """Starting a new generation should clear the inline preview."""

        audio, remaining, area, area_original, lego_part, status = clear_inline_result_preview()

        self.assertEqual(audio["value"], None)
        self.assertEqual(remaining["value"], None)
        self.assertFalse(remaining["visible"])
        self.assertEqual(area["value"], None)
        self.assertFalse(area["visible"])
        self.assertEqual(area_original["value"], None)
        self.assertFalse(area_original["visible"])
        self.assertEqual(lego_part["value"], None)
        self.assertFalse(lego_part["visible"])
        self.assertEqual(status, "")

    def test_prepare_inline_result_preview_shows_extract_progress(self):
        """Starting Extract should show progress in the inline latest-result preview."""

        audio, remaining, area, area_original, lego_part, status = (
            prepare_inline_result_preview("extract")
        )

        self.assertEqual(audio["value"], None)
        self.assertEqual(audio["label"], "Extracted Audio")
        self.assertEqual(remaining["value"], None)
        self.assertTrue(remaining["visible"])
        self.assertFalse(area["visible"])
        self.assertFalse(area_original["visible"])
        self.assertFalse(lego_part["visible"])
        self.assertEqual(status, t("messages.extract_stem_processing"))

    def test_prepare_inline_result_preview_keeps_non_extract_behavior(self):
        """Starting non-Extract generation should keep the existing blank status."""

        audio, remaining, area, area_original, lego_part, status = (
            prepare_inline_result_preview("text2music")
        )

        self.assertEqual(audio["value"], None)
        self.assertEqual(remaining["value"], None)
        self.assertFalse(remaining["visible"])
        self.assertFalse(area["visible"])
        self.assertFalse(area_original["visible"])
        self.assertFalse(lego_part["visible"])
        self.assertEqual(status, "")

    def test_inline_result_preview_from_generation_outputs_copies_status(self):
        """Streamed generation status should be mirrored into the inline preview."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_remaining.mp3"]

        audio, remaining, area, area_original, lego_part, status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertEqual(audio, "out_0")
        self.assertEqual(remaining["value"], "/tmp/song_remaining.mp3")
        self.assertEqual(remaining["label"], "Original Input")
        self.assertTrue(remaining["visible"])
        self.assertFalse(area["visible"])
        self.assertFalse(area_original["visible"])
        self.assertFalse(lego_part["visible"])
        self.assertEqual(status, "out_18")

    def test_inline_result_preview_prefers_original_source_audio(self):
        """Source-edit results should show original input beside Sample 1."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_remaining.mp3", "/tmp/source_audio.wav"]

        _audio, original, _area, _area_original, _lego_part, _status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertEqual(original["value"], "/tmp/source_audio.wav")
        self.assertEqual(original["label"], "Original Input")
        self.assertTrue(original["visible"])

    def test_inline_result_preview_shows_latest_repainted_area(self):
        """Source-edit outputs should show generated and original edited-area clips."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = [
            "/tmp/song_latest_repainted_area.wav",
            "/tmp/song_latest_repainted_area_original.wav",
        ]

        _audio, _remaining, area, area_original, lego_part, _status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertEqual(area["value"], "/tmp/song_latest_repainted_area.wav")
        self.assertEqual(area["label"], "Latest Repainted Area")
        self.assertTrue(area["visible"])
        self.assertEqual(
            area_original["value"],
            "/tmp/song_latest_repainted_area_original.wav",
        )
        self.assertEqual(area_original["label"], "Latest Repainted Area Original")
        self.assertTrue(area_original["visible"])
        self.assertFalse(lego_part["visible"])

    def test_inline_result_preview_shows_latest_remixed_area_label(self):
        """Remix outputs should not use repaint labels for latest-area clips."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = [
            "/tmp/song_latest_remixed_area.wav",
            "/tmp/song_latest_remixed_area_original.wav",
        ]

        _audio, _remaining, area, area_original, lego_part, _status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertEqual(area["label"], "Latest Remixed Area")
        self.assertEqual(area_original["label"], "Latest Remixed Area Original")
        self.assertTrue(area["visible"])
        self.assertTrue(area_original["visible"])
        self.assertFalse(lego_part["visible"])

    def test_inline_result_preview_shows_latest_lego_area_label(self):
        """Lego edited-area clips should use Lego labels."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = [
            "/tmp/song_latest_lego_area.wav",
            "/tmp/song_latest_lego_area_original.wav",
        ]

        _audio, _remaining, area, area_original, lego_part, _status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertEqual(area["label"], "Latest LEGO Area")
        self.assertEqual(area_original["label"], "Latest LEGO Area Original")
        self.assertTrue(area["visible"])
        self.assertTrue(area_original["visible"])
        self.assertFalse(lego_part["visible"])

    def test_inline_result_preview_shows_full_lego_generated_track_separately(self):
        """The raw full Lego layer should show in its own third row."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_lego_generated_track.wav"]

        _audio, _remaining, area, area_original, lego_part, _status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertFalse(area["visible"])
        self.assertFalse(area_original["visible"])
        self.assertEqual(lego_part["value"], "/tmp/song_lego_generated_track.wav")
        self.assertEqual(lego_part["label"], "Latest LEGO Part Only")
        self.assertTrue(lego_part["visible"])

    def test_inline_result_preview_from_generation_outputs_preserves_skip_status(self):
        """No-op backend status updates should not erase the inline status."""

        outputs = tuple(["sample.wav", *["unused"] * 15, [], "unused", gr.skip()])

        audio, remaining, area, area_original, lego_part, status = (
            inline_result_preview_from_generation_outputs(outputs)
        )

        self.assertEqual(audio, "sample.wav")
        self.assertFalse(remaining["visible"])
        self.assertFalse(area["visible"])
        self.assertFalse(area_original["visible"])
        self.assertFalse(lego_part["visible"])
        self.assertEqual(status, gr.skip())

    def test_append_inline_result_preview_extends_generation_outputs(self):
        """Generation wrapper outputs should include inline audio and status at the end."""

        outputs = list(f"out_{index}" for index in range(63))
        outputs[16] = ["/tmp/song_remaining.flac"]
        outputs = tuple(outputs)

        result = append_inline_result_preview(outputs)

        self.assertEqual(result[:-6], outputs)
        self.assertEqual(result[-6], "out_0")
        self.assertEqual(result[-5]["value"], "/tmp/song_remaining.flac")
        self.assertEqual(result[-5]["label"], "Original Input")
        self.assertFalse(result[-4]["visible"])
        self.assertFalse(result[-3]["visible"])
        self.assertFalse(result[-2]["visible"])
        self.assertEqual(result[-1], "out_18")

    def test_sync_inline_result_preview_mirrors_first_sample_and_status(self):
        """Completed generation should copy Sample 1 and status into the preview."""

        audio, remaining, area, area_original, lego_part, status = sync_inline_result_preview(
            "sample_1.wav",
            ["/tmp/sample_1_remaining.wav"],
            "Generation Complete",
        )

        self.assertEqual(audio, "sample_1.wav")
        self.assertEqual(remaining["value"], "/tmp/sample_1_remaining.wav")
        self.assertEqual(remaining["label"], "Original Input")
        self.assertTrue(remaining["visible"])
        self.assertFalse(area["visible"])
        self.assertFalse(area_original["visible"])
        self.assertFalse(lego_part["visible"])
        self.assertEqual(status, "Generation Complete")


if __name__ == "__main__":
    unittest.main()
