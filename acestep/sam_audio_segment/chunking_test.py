"""Tests for SAM-Audio chunk overlap stitching."""

import unittest

import torch

from acestep.sam_audio_segment.chunking import (
    iter_audio_chunks,
    overlap_add_chunks,
    should_process_chunked,
)


class TestSamAudioChunking(unittest.TestCase):
    """Verify long-audio chunk coverage and reconstruction helpers."""

    def test_iter_chunks_covers_full_audio_with_overlap(self):
        """Chunk windows should cover the whole source without gaps."""

        audio = torch.zeros(1, 100)
        chunks = list(iter_audio_chunks(audio, 10, 4.0, 1.0))

        self.assertEqual(0, chunks[0].start)
        self.assertEqual(100, chunks[-1].end)
        self.assertTrue(all(left.end > right.start for left, right in zip(chunks, chunks[1:])))

    def test_overlap_add_preserves_length(self):
        """Stitched chunks should keep the original sample count."""

        chunks = [
            (0, 6, torch.ones(1, 6)),
            (4, 10, torch.ones(1, 6) * 2.0),
        ]
        stitched = overlap_add_chunks(chunks, total_samples=10, overlap_samples=2)

        self.assertEqual((1, 10), tuple(stitched.shape))
        self.assertTrue(torch.isfinite(stitched).all())

    def test_should_process_chunked_only_for_long_audio(self):
        """Audio shorter than the chunk window should remain single-pass."""

        self.assertFalse(should_process_chunked(torch.zeros(1, 10), 10, 2.0))
        self.assertTrue(should_process_chunked(torch.zeros(1, 30), 10, 2.0))


if __name__ == "__main__":
    unittest.main()
