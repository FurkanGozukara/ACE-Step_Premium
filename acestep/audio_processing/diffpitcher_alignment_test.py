"""Tests for DiffPitcher template pitch alignment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from acestep.audio_processing.diffpitcher_alignment import align_reference_f0


class DiffPitcherAlignmentTests(unittest.TestCase):
    """Verify exact and coarse reference F0 alignment paths."""

    def test_align_reference_f0_uses_exact_dtw_for_short_inputs(self) -> None:
        """Short references should use full-resolution DTW alignment."""

        source_features = np.arange(6, dtype=np.float32)[None, :]
        reference_features = np.arange(6, dtype=np.float32)[None, :]
        reference_f0 = np.arange(10, 70, 10, dtype=np.float32)

        with patch(
            "acestep.audio_processing.diffpitcher_alignment._mfcc_features",
            side_effect=[source_features, reference_features],
        ):
            aligned, mode = align_reference_f0(
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                reference_f0,
                6,
            )

        self.assertEqual("exact_dtw", mode)
        np.testing.assert_allclose(reference_f0, aligned)

    def test_align_reference_f0_uses_coarse_dtw_for_long_inputs(self) -> None:
        """Long references should avoid nearest-only fallback and use coarse DTW."""

        source_features = np.linspace(0, 1, 2200, dtype=np.float32)[None, :]
        reference_features = np.linspace(0, 1, 2200, dtype=np.float32)[None, :]
        reference_f0 = np.arange(2200, dtype=np.float32)

        with patch(
            "acestep.audio_processing.diffpitcher_alignment._mfcc_features",
            side_effect=[source_features, reference_features],
        ):
            aligned, mode = align_reference_f0(
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                reference_f0,
                2200,
            )

        self.assertEqual("coarse_dtw", mode)
        self.assertEqual((2200,), aligned.shape)
        self.assertLess(abs(float(aligned[0])), 1.0)
        self.assertLess(abs(float(aligned[-1]) - 2199.0), 5.0)


if __name__ == "__main__":
    unittest.main()
