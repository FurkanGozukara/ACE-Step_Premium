"""Tests for SAM-Audio seed resolution."""

import unittest

from acestep.sam_audio_segment.seed import resolve_runtime_seed
from acestep.sam_audio_segment.settings import SamAudioSettings


class TestSamAudioSeed(unittest.TestCase):
    """Verify deterministic and random seed handling."""

    def test_deterministic_seed_is_preserved(self):
        """Random seed disabled should keep the UI seed value."""

        settings = SamAudioSettings(seed=99, random_seed=False)

        self.assertEqual(99, resolve_runtime_seed(settings).seed)

    def test_random_seed_replaces_ui_seed(self):
        """Random seed enabled should choose a valid per-run seed."""

        settings = SamAudioSettings(seed=99, random_seed=True)
        resolved = resolve_runtime_seed(settings)

        self.assertTrue(resolved.random_seed)
        self.assertGreaterEqual(resolved.seed, 1)
        self.assertNotEqual(99, resolved.seed)


if __name__ == "__main__":
    unittest.main()
