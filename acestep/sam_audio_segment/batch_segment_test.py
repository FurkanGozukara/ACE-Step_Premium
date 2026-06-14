"""Tests for SAM-Audio Batch Segment prompt parsing."""

import unittest

from acestep.sam_audio_segment.batch_segment import (
    batch_segment_prompts,
    batch_segment_suffix,
    is_batch_segment_active,
    parse_batch_segment_prompts,
)
from acestep.sam_audio_segment.settings import SamAudioSettings


class TestSamAudioBatchSegment(unittest.TestCase):
    """Verify Batch Segment prompt parsing and output suffixes."""

    def test_semicolon_prompts_are_trimmed_normalized_and_suffixed(self) -> None:
        """Prompt text should be model-ready and filename-safe."""

        prompts = parse_batch_segment_prompts(" Vocals ; electric guitar ; bass  ")

        self.assertEqual(
            ["vocals", "electric guitar", "bass"],
            [item.text for item in prompts],
        )
        self.assertEqual(
            ["vocals", "electric_guitar", "bass"],
            [item.suffix for item in prompts],
        )

    def test_duplicate_suffixes_are_made_unique(self) -> None:
        """Equivalent prompts should not overwrite output files."""

        prompts = parse_batch_segment_prompts("lead vocal;lead-vocal;lead vocal")

        self.assertEqual(
            ["lead_vocal", "lead_vocal_2", "lead_vocal_3"],
            [item.suffix for item in prompts],
        )

    def test_empty_enabled_prompt_list_raises_clear_error(self) -> None:
        """Enabled Batch Segment requires at least one Custom Prompt value."""

        with self.assertRaisesRegex(ValueError, "semicolon-separated prompts"):
            batch_segment_prompts(
                SamAudioSettings(batch_segment=True, custom_prompt=" ; ")
            )

    def test_empty_quick_prompt_selection_raises_clear_error(self) -> None:
        """Batch Segment should require at least one prompt source."""

        with self.assertRaisesRegex(ValueError, "Quick Prompt"):
            batch_segment_prompts(
                SamAudioSettings(batch_segment=True, prompt_preset=())
            )

    def test_quick_prompt_selections_are_used_when_custom_prompt_is_empty(self) -> None:
        """Batch Segment can fan out selected Quick Prompt presets."""

        prompts = batch_segment_prompts(
            SamAudioSettings(
                batch_segment=True,
                prompt_preset=("vocals", "electric guitar", "bass"),
            )
        )

        self.assertEqual(
            ["vocals", "electric guitar", "bass"],
            [item.text for item in prompts],
        )
        self.assertEqual(
            ["vocals", "electric_guitar", "bass"],
            [item.suffix for item in prompts],
        )

    def test_multiple_quick_prompts_enable_batch_segment_without_checkbox(self) -> None:
        """Multiple Quick Prompt selections should behave like Batch Segment."""

        settings = SamAudioSettings(
            batch_segment=False,
            prompt_preset=("vocals", "electric guitar", "bass"),
        )
        prompts = batch_segment_prompts(settings)

        self.assertTrue(is_batch_segment_active(settings))
        self.assertEqual(
            ["vocals", "electric guitar", "bass"],
            [item.text for item in prompts],
        )

    def test_single_quick_prompt_without_checkbox_stays_single_run(self) -> None:
        """A single Quick Prompt should stay a normal SAM run."""

        settings = SamAudioSettings(batch_segment=False, prompt_preset=("vocals",))

        self.assertFalse(is_batch_segment_active(settings))
        self.assertEqual([], batch_segment_prompts(settings))

    def test_custom_prompt_blocks_auto_quick_prompt_batch_segment(self) -> None:
        """Typed Custom Prompt remains the explicit prompt source."""

        settings = SamAudioSettings(
            batch_segment=False,
            prompt_preset=("vocals", "bass"),
            custom_prompt="guitar",
        )

        self.assertFalse(is_batch_segment_active(settings))
        self.assertEqual([], batch_segment_prompts(settings))

    def test_custom_prompt_overrides_quick_prompt_selections(self) -> None:
        """Typed prompts should remain the explicit Batch Segment source."""

        prompts = batch_segment_prompts(
            SamAudioSettings(
                batch_segment=True,
                prompt_preset=("vocals", "bass"),
                custom_prompt="guitar;drums",
            )
        )

        self.assertEqual(["guitar", "drums"], [item.text for item in prompts])

    def test_batch_segment_disabled_returns_empty_prompt_list(self) -> None:
        """Normal SAM-Audio runs should ignore semicolons unless enabled."""

        prompts = batch_segment_prompts(
            SamAudioSettings(batch_segment=False, custom_prompt="vocals;guitar")
        )

        self.assertEqual([], prompts)

    def test_suffix_has_fallback_for_symbols_only(self) -> None:
        """Filesystem suffixes should always be usable."""

        self.assertEqual("segment", batch_segment_suffix("!!!"))


if __name__ == "__main__":
    unittest.main()
