"""Tests for shared training tensor-folder browse actions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring import training_tensor_browse


class TrainingTensorBrowseTests(unittest.TestCase):
    """Verify tensor-folder browse callbacks handle selection states."""

    def test_reselecting_current_folder_loads_dataset(self) -> None:
        """Selecting the current folder should update the textbox and load."""

        selected = "C:\\ace_step_training_tutorial\\tensor_of_2pac"
        with patch.object(
            training_tensor_browse,
            "select_optional_folder_path",
            return_value=selected,
        ), patch.object(
            training_tensor_browse.train_h,
            "load_training_dataset",
            return_value="loaded",
        ) as load_dataset:
            textbox_update, status = training_tensor_browse.browse_and_load_training_dataset(
                selected
            )

        self.assertEqual(selected, textbox_update["value"])
        self.assertEqual("loaded", status)
        load_dataset.assert_called_once_with(selected)

    def test_cancel_keeps_textbox_and_reports_no_selection(self) -> None:
        """Canceling the native dialog should leave the textbox unchanged."""

        with patch.object(
            training_tensor_browse,
            "select_optional_folder_path",
            return_value=None,
        ), patch.object(
            training_tensor_browse.train_h,
            "load_training_dataset",
        ) as load_dataset:
            textbox_update, status = training_tensor_browse.browse_and_load_training_dataset(
                "C:\\temp"
            )

        self.assertNotIn("value", textbox_update)
        self.assertEqual("No tensor folder selected.", status)
        load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
