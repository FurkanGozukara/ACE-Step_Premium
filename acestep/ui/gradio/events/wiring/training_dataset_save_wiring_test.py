"""Tests for dataset save and saved-JSON reload wiring."""

from __future__ import annotations

import unittest
from unittest.mock import ANY, patch

import gradio as gr

from acestep.ui.gradio.events.wiring import training_dataset_save_wiring


class TrainingDatasetSaveWiringTests(unittest.TestCase):
    """Verify dataset save callbacks update the load path and load status."""

    def test_successful_save_loads_saved_path(self) -> None:
        """A successful save should immediately load the saved dataset JSON."""

        builder_state = object()
        saved_path = "/workspace/ACE-Step_Premium/datasets/my_lora.json"
        save_update = gr.update(value=saved_path)
        with patch.object(
            training_dataset_save_wiring.train_h,
            "save_dataset",
            return_value=("\u2705 Dataset saved", save_update),
        ) as save_dataset, patch.object(
            training_dataset_save_wiring.train_h,
            "load_existing_dataset_for_preprocess",
            return_value=("loaded", "table"),
        ) as load_dataset:
            result = training_dataset_save_wiring._save_dataset_and_load_saved_path(
                "datasets/my_lora",
                "my_lora",
                "ohwx",
                "prepend",
                False,
                0,
                builder_state,
            )

        self.assertEqual("\u2705 Dataset saved", result[0])
        self.assertEqual(save_update, result[1])
        self.assertEqual(saved_path, result[2]["value"])
        self.assertEqual(("loaded", "table"), result[3:])
        save_dataset.assert_called_once_with(
            "datasets/my_lora",
            "my_lora",
            builder_state,
            custom_tag="ohwx",
            tag_position="prepend",
            all_instrumental=False,
            genre_ratio=0,
        )
        load_dataset.assert_called_once_with(
            saved_path,
            builder_state,
        )

    def test_failed_save_does_not_load_stale_path(self) -> None:
        """A failed save should not trigger a load using any previous path."""

        with patch.object(
            training_dataset_save_wiring.train_h,
            "save_dataset",
            return_value=("\u274c Failed to save dataset", gr.update()),
        ), patch.object(
            training_dataset_save_wiring.train_h,
            "load_existing_dataset_for_preprocess",
        ) as load_dataset:
            result = training_dataset_save_wiring._save_dataset_and_load_saved_path(
                "bad.json",
                "my_lora",
                "ohwx",
                "prepend",
                False,
                0,
                object(),
            )

        self.assertEqual("\u274c Failed to save dataset", result[0])
        self.assertNotIn("value", result[2])
        self.assertEqual("Save did not complete; dataset was not loaded.", result[3])
        load_dataset.assert_not_called()

    def test_runpod_browse_save_uses_current_path_when_dialog_is_unavailable(self) -> None:
        """Browse-save should still save on hosted systems without native dialogs."""

        current_path = "/workspace/ACE-Step_Premium/datasets/my_lora_dataset.json"
        with patch.object(
            training_dataset_save_wiring,
            "select_json_save_path",
            return_value=current_path,
        ), patch.object(
            training_dataset_save_wiring,
            "is_dialog_available",
            return_value=False,
        ), patch.object(
            training_dataset_save_wiring,
            "_save_dataset_and_load_saved_path",
            return_value=("saved",),
        ) as save_and_load:
            result = training_dataset_save_wiring._browse_save_dataset_and_load_saved_path(
                current_path,
                "my_lora",
                "ohwx",
                "prepend",
                False,
                0,
                object(),
            )

        self.assertEqual(("saved",), result)
        save_and_load.assert_called_once_with(
            current_path,
            "my_lora",
            "ohwx",
            "prepend",
            False,
            0,
            ANY,
        )


if __name__ == "__main__":
    unittest.main()
