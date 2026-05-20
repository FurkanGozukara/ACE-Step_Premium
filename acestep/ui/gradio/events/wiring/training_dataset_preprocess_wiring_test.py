"""Tests for training dataset preprocess browse wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring import training_dataset_preprocess_wiring


class TrainingDatasetPreprocessWiringTests(unittest.TestCase):
    """Verify Step 5 dataset JSON browse/load callback behavior."""

    def test_reselecting_current_json_loads_dataset(self) -> None:
        """Selecting the current JSON path should still load it."""

        selected = "C:\\ace_step_training_tutorial\\tupac_fst_dataset.json"
        builder_state = object()
        with patch.object(
            training_dataset_preprocess_wiring,
            "select_optional_json_file_path",
            return_value=selected,
        ), patch.object(
            training_dataset_preprocess_wiring.train_h,
            "load_existing_dataset_for_preprocess",
            return_value=("loaded",),
        ) as load_dataset:
            result = training_dataset_preprocess_wiring._browse_and_load_dataset_json(
                selected,
                builder_state,
            )

        self.assertEqual(selected, result[0]["value"])
        self.assertEqual("loaded", result[1])
        load_dataset.assert_called_once_with(selected, builder_state)

    def test_cancel_keeps_textbox_and_reports_no_selection(self) -> None:
        """Canceling the native dialog should leave Step 5 state unchanged."""

        with patch.object(
            training_dataset_preprocess_wiring,
            "select_optional_json_file_path",
            return_value=None,
        ), patch.object(
            training_dataset_preprocess_wiring.train_h,
            "load_existing_dataset_for_preprocess",
        ) as load_dataset:
            result = training_dataset_preprocess_wiring._browse_and_load_dataset_json(
                "C:\\temp\\dataset.json",
                object(),
            )

        self.assertNotIn("value", result[0])
        self.assertEqual("No dataset JSON selected.", result[1])
        load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
