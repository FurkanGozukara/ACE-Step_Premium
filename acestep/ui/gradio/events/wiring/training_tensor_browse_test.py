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

    def test_lora_browse_sets_loaded_state_after_successful_load(self) -> None:
        """LoRA browse should expose active dataset state only after success."""

        selected = "C:\\ace_step_training_tutorial\\tensor_of_2pac"
        with patch.object(
            training_tensor_browse,
            "select_optional_folder_path",
            return_value=selected,
        ), patch.object(
            training_tensor_browse.train_h,
            "load_training_dataset",
            return_value="Loaded preprocessed dataset: test",
        ):
            textbox_update, status, loaded_dir = (
                training_tensor_browse.browse_and_load_lora_training_dataset(selected)
            )

        self.assertEqual(selected, textbox_update["value"])
        self.assertIn("Loaded preprocessed dataset", status)
        self.assertEqual(selected, loaded_dir)

    def test_lora_load_clears_loaded_state_after_failed_load(self) -> None:
        """LoRA load should keep validation inactive when loading fails."""

        status, loaded_dir = training_tensor_browse.load_lora_training_dataset_with_state(
            ""
        )

        self.assertIn("Please enter", status)
        self.assertEqual("", loaded_dir)

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

    def test_lora_cancel_clears_loaded_state(self) -> None:
        """Canceling LoRA browse should leave no active loaded dataset."""

        with patch.object(
            training_tensor_browse,
            "select_optional_folder_path",
            return_value=None,
        ), patch.object(
            training_tensor_browse.train_h,
            "load_training_dataset",
        ) as load_dataset:
            textbox_update, status, loaded_dir = (
                training_tensor_browse.browse_and_load_lora_training_dataset("C:\\temp")
            )

        self.assertNotIn("value", textbox_update)
        self.assertEqual("No tensor folder selected.", status)
        self.assertEqual("", loaded_dir)
        load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
