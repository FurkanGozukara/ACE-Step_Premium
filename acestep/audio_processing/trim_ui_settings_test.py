"""Tests for shared Auto-Editor trim UI value parsing."""

from __future__ import annotations

import unittest

from acestep.audio_processing.trim_ui_settings import trim_settings_from_ui_values


class TrimUiSettingsTests(unittest.TestCase):
    """Verify shared Auto-Editor trim controls normalize safely."""

    def test_trim_settings_from_ui_values_clamps_shared_controls(self) -> None:
        """Ordered Audio Processing trim values should produce safe trim settings."""

        settings = trim_settings_from_ui_values([-120.0, 20.0, 999, -5])

        self.assertEqual(-100.0, settings.threshold_db)
        self.assertEqual(5.0, settings.margin_seconds)
        self.assertEqual(300, settings.mincut)
        self.assertEqual(0, settings.minclip)


if __name__ == "__main__":
    unittest.main()
