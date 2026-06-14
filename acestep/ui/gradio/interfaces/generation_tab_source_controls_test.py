"""Tests for source-audio generation controls."""

import unittest

from acestep.constants import DEFAULT_EXTRACT_TRACK_NAME
from acestep.ui.gradio.interfaces.generation_tab_source_controls import (
    _default_track_name,
)


class GenerationTabSourceControlsTests(unittest.TestCase):
    """Verify source-audio control defaults."""

    def test_default_track_name_is_vocals(self):
        """Track Name should default to vocals, not the first TRACK_NAMES item."""

        self.assertEqual(DEFAULT_EXTRACT_TRACK_NAME, _default_track_name({}))

    def test_default_track_name_preserves_valid_init_param(self):
        """Startup params should still be able to preselect a valid track."""

        self.assertEqual("guitar", _default_track_name({"track_name": " Guitar "}))


if __name__ == "__main__":
    unittest.main()
