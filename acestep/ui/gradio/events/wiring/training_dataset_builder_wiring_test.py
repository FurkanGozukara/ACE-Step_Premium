"""Tests for training dataset-builder wiring helpers."""

import unittest

from acestep.ui.gradio.events.wiring.training_dataset_status import (
    append_preview_refresh_status,
)


class TrainingDatasetBuilderWiringTests(unittest.TestCase):
    """Tests for auto-label status formatting."""

    def test_success_status_appends_preview_refresh(self):
        """Successful auto-label status should mention the refreshed preview."""

        status = append_preview_refresh_status("Labeled")

        self.assertIn("Labeled", status)
        self.assertIn("Preview refreshed", status)

    def test_failure_status_does_not_append_preview_refresh(self):
        """Failure status should not add a misleading preview success line."""

        status = append_preview_refresh_status("ERROR: Failed to initialize")

        self.assertEqual("ERROR: Failed to initialize", status)


if __name__ == "__main__":
    unittest.main()
