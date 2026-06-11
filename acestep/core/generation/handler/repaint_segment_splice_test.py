"""Tests for local repaint segment waveform insertion."""

import unittest

import torch

from acestep.core.generation.handler.repaint_segment_splice import (
    apply_repaint_segment_splice,
)


class RepaintSegmentSpliceTests(unittest.TestCase):
    """Verify locally generated repaint audio is inserted into source audio."""

    def test_inserts_segment_inside_requested_region(self):
        """Only the requested waveform range should be replaced."""
        source = torch.zeros(1, 2, 10)
        segment = torch.ones(1, 2, 4)

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.3],
            repainting_ends=[0.7],
            sample_rate=10,
        )

        self.assertTrue(torch.equal(result[..., :3], torch.zeros(1, 2, 3)))
        self.assertTrue(torch.equal(result[..., 3:7], torch.ones(1, 2, 4)))
        self.assertTrue(torch.equal(result[..., 7:], torch.zeros(1, 2, 3)))

    def test_short_segment_removes_unused_selected_tail(self):
        """Short generated segments should shrink output instead of padding silence."""
        source = torch.full((1, 2, 10), 9.0)
        segment = torch.ones(1, 2, 2)

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.6],
            sample_rate=10,
        )

        self.assertEqual((1, 2, 8), tuple(result.shape))
        self.assertTrue(torch.equal(result[..., :2], torch.full((1, 2, 2), 9.0)))
        self.assertTrue(torch.equal(result[..., 2:4], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(result[..., 4:], torch.full((1, 2, 4), 9.0)))

    def test_trailing_generated_silence_removes_unused_selected_tail(self):
        """Trailing generated silence should be removed from the replacement."""
        source = torch.full((1, 2, 10), 9.0)
        segment = torch.zeros(1, 2, 4)
        segment[..., :2] = 1.0

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.6],
            sample_rate=10,
        )

        self.assertEqual((1, 2, 8), tuple(result.shape))
        self.assertTrue(torch.equal(result[..., :2], torch.full((1, 2, 2), 9.0)))
        self.assertTrue(torch.equal(result[..., 2:4], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(result[..., 4:], torch.full((1, 2, 4), 9.0)))

    def test_trailing_low_level_noise_removes_unused_selected_tail(self):
        """Low-RMS decoded noise should be treated as trailing silence."""
        source = torch.full((1, 2, 10), 9.0)
        segment = torch.full((1, 2, 4), 0.001)
        segment[..., :2] = 1.0

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.6],
            sample_rate=10,
        )

        self.assertEqual((1, 2, 8), tuple(result.shape))
        self.assertTrue(torch.equal(result[..., 2:4], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(result[..., 4:], torch.full((1, 2, 4), 9.0)))

    def test_replacement_strength_zero_preserves_selected_source_audio(self):
        """Strength zero should use selected source audio instead of generated audio."""
        source = torch.arange(10, dtype=torch.float32).view(1, 1, 10).expand(1, 2, 10)
        segment = torch.full((1, 2, 4), 99.0)

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.6],
            sample_rate=10,
            replacement_strength=0.0,
        )

        self.assertEqual((1, 2, 10), tuple(result.shape))
        torch.testing.assert_close(result, source)

    def test_replacement_strength_blends_selected_source_audio(self):
        """Intermediate strength should audibly blend source and generated audio."""
        source = torch.full((1, 2, 10), 9.0)
        segment = torch.ones(1, 2, 4)

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.6],
            sample_rate=10,
            replacement_strength=0.5,
        )

        self.assertEqual((1, 2, 10), tuple(result.shape))
        self.assertTrue(torch.equal(result[..., 2:6], torch.full((1, 2, 4), 5.0)))

    def test_long_segment_expands_output_without_overwriting_source_tail(self):
        """Long generated segments should expand output and push source tail later."""
        source = torch.arange(10, dtype=torch.float32).view(1, 1, 10).expand(1, 2, 10)
        segment = torch.full((1, 2, 5), 99.0)

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.4],
            sample_rate=10,
        )

        self.assertEqual((1, 2, 13), tuple(result.shape))
        torch.testing.assert_close(result[..., :2], source[..., :2])
        self.assertTrue(torch.equal(result[..., 2:7], torch.full((1, 2, 5), 99.0)))
        torch.testing.assert_close(result[..., 7:], source[..., 4:])


if __name__ == "__main__":
    unittest.main()
