"""Tests for training dataset preprocess browse wiring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.ui.gradio.events.wiring import training_dataset_preprocess_wiring
from acestep.ui.gradio.events.wiring.training_dataset_load_outputs import (
    DATASET_LOAD_SHARED_OUTPUT_KEYS,
)


class TrainingDatasetPreprocessWiringTests(unittest.TestCase):
    """Verify Step 5 dataset JSON browse/load callback behavior."""

    def test_load_button_calls_loader_directly(self) -> None:
        """Load button wiring should update load status from the loader callback."""

        button = _FakeComponent()
        training_section = _training_section(button)
        context = SimpleNamespace(training_section=training_section)

        training_dataset_preprocess_wiring.register_training_dataset_load_handler(
            context,
            button_key="load_existing_dataset_btn",
            path_key="load_existing_dataset_path",
            status_key="load_existing_status",
        )

        self.assertIs(
            training_dataset_preprocess_wiring.train_h.load_existing_dataset_for_preprocess,
            button.click_kwargs["fn"],
        )
        self.assertEqual(
            [
                training_section["load_existing_dataset_path"],
                training_section["dataset_builder_state"],
            ],
            button.click_kwargs["inputs"],
        )
        self.assertEqual(
            training_section["load_existing_status"],
            button.click_kwargs["outputs"][0],
        )

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

    def test_cancel_preprocess_click_runs_backend_directly(self) -> None:
        """Preprocess cancel should be a direct unqueued backend event."""

        cancel_button = _FakeComponent()
        section = _preprocess_section(cancel_button)
        context = SimpleNamespace(training_section=section, dit_handler=object())

        training_dataset_preprocess_wiring.register_training_preprocess_handler(context)

        click_kwargs = cancel_button.click_kwargs
        self.assertIs(
            training_dataset_preprocess_wiring.request_preprocess_cancel_from_ui,
            click_kwargs["fn"],
        )
        self.assertIsNone(click_kwargs["inputs"])
        self.assertEqual([section["preprocess_progress"]], click_kwargs["outputs"])
        self.assertFalse(click_kwargs["queue"])
        self.assertIn("confirm", click_kwargs["js"])

    def test_preprocess_click_reads_current_dataset_settings(self) -> None:
        """Preprocess should not depend on pending dataset-setting change events."""

        section = _preprocess_section(_FakeComponent())
        preprocess_button = section["preprocess_btn"]
        context = SimpleNamespace(training_section=section, dit_handler=object())

        training_dataset_preprocess_wiring.register_training_preprocess_handler(context)

        self.assertEqual(
            [
                section["custom_tag"],
                section["tag_position"],
                section["all_instrumental"],
                section["genre_ratio"],
                section["use_only_custom_trigger"],
            ],
            preprocess_button.click_kwargs["inputs"][-5:],
        )
        self.assertEqual(
            section["preprocess_debug_text"],
            preprocess_button.click_kwargs["inputs"][2],
        )


class _FakeEvent:
    """Minimal event object for asserting chained Gradio wiring."""

    def then(self, **_kwargs):
        """Record no details; the test only needs the click callback."""

        return self


class _FakeComponent:
    """Minimal Gradio component test double."""

    def __init__(self) -> None:
        """Create an empty click-call recorder."""

        self.click_kwargs = None

    def click(self, **kwargs):
        """Record click registration arguments."""

        self.click_kwargs = kwargs
        return _FakeEvent()


def _training_section(button):
    """Return a component map containing all dataset-load output keys."""

    section = {
        "load_existing_dataset_btn": button,
        "load_existing_dataset_path": object(),
        "load_existing_status": object(),
        "dataset_builder_state": object(),
        "has_raw_lyrics_state": object(),
        "raw_lyrics_display": object(),
    }
    for key in DATASET_LOAD_SHARED_OUTPUT_KEYS:
        section.setdefault(key, object())
    return section


def _preprocess_section(cancel_button):
    """Return a component map for preprocess wiring tests."""

    return {
        "preprocess_output_dir_browse_btn": _FakeComponent(),
        "preprocess_output_dir": object(),
        "preprocess_btn": _FakeComponent(),
        "preprocess_mode": object(),
        "preprocess_debug_text": object(),
        "dataset_builder_state": object(),
        "dataset_model_config": object(),
        "dataset_vram_preset": object(),
        "preprocess_subprocess": object(),
        "custom_tag": object(),
        "tag_position": object(),
        "all_instrumental": object(),
        "genre_ratio": object(),
        "use_only_custom_trigger": object(),
        "preprocess_progress": object(),
        "cancel_preprocess_btn": cancel_button,
    }


if __name__ == "__main__":
    unittest.main()
