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

    def test_trims_or_pads_segment_to_region_length(self):
        """Generated segment length mismatches should not change source length."""
        source = torch.zeros(1, 2, 10)
        segment = torch.ones(1, 2, 2)

        result = apply_repaint_segment_splice(
            pred_segments=segment,
            src_wavs=source,
            repainting_starts=[0.2],
            repainting_ends=[0.6],
            sample_rate=10,
        )

        self.assertEqual((1, 2, 10), tuple(result.shape))
        self.assertTrue(torch.equal(result[..., 2:4], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(result[..., 4:6], torch.zeros(1, 2, 2)))


if __name__ == "__main__":
    unittest.main()
