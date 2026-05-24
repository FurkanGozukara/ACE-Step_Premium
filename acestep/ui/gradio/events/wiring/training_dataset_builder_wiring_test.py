"""Tests for training dataset-builder wiring helpers."""

import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

from acestep.ui.gradio.events.wiring.context import TrainingWiringContext
from acestep.ui.gradio.events.wiring.training_dataset_builder_wiring import (
    register_training_dataset_builder_handlers,
)
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

    def test_cancel_auto_label_click_runs_backend_directly(self):
        """Cancel should not depend on a delayed chained event."""

        with gr.Blocks() as demo:
            training_section = _minimal_training_section()
            context = TrainingWiringContext(
                demo=demo,
                dit_handler=MagicMock(),
                llm_handler=MagicMock(),
                training_section=training_section,
            )
            with patch(
                "acestep.ui.gradio.events.wiring.training_dataset_builder_wiring."
                "register_training_dataset_save_handlers"
            ):
                register_training_dataset_builder_handlers(context)

        cancel_id = training_section["cancel_auto_label_btn"]._id
        progress_id = training_section["label_progress"]._id
        cancel_dependencies = [
            dependency
            for dependency in demo.config["dependencies"]
            if dependency["targets"] == [(cancel_id, "click")]
            and dependency["backend_fn"]
        ]

        self.assertEqual(1, len(cancel_dependencies))
        dependency = cancel_dependencies[0]
        self.assertFalse(dependency["queue"])
        self.assertIn("confirm", dependency["js"])
        self.assertEqual([], dependency["inputs"])
        self.assertEqual([progress_id], dependency["outputs"])

    def test_auto_label_click_reads_current_dataset_settings(self):
        """Auto-label should not depend on pending dataset-setting change events."""

        with gr.Blocks() as demo:
            training_section = _minimal_training_section()
            context = TrainingWiringContext(
                demo=demo,
                dit_handler=MagicMock(),
                llm_handler=MagicMock(),
                training_section=training_section,
            )
            with patch(
                "acestep.ui.gradio.events.wiring.training_dataset_builder_wiring."
                "register_training_dataset_save_handlers"
            ):
                register_training_dataset_builder_handlers(context)

        auto_label_id = training_section["auto_label_btn"]._id
        setting_ids = [
            training_section["custom_tag"]._id,
            training_section["tag_position"]._id,
            training_section["all_instrumental"]._id,
            training_section["genre_ratio"]._id,
        ]
        auto_label_dependencies = [
            dependency
            for dependency in demo.config["dependencies"]
            if dependency["targets"] == [(auto_label_id, "click")]
            and dependency["backend_fn"]
        ]

        self.assertEqual(1, len(auto_label_dependencies))
        for setting_id in setting_ids:
            self.assertIn(setting_id, auto_label_dependencies[0]["inputs"])


def _minimal_training_section() -> dict[str, object]:
    """Build enough Gradio controls to register dataset-builder handlers."""

    section: dict[str, object] = {
        "audio_files_table": gr.Dataframe(),
        "dataset_builder_state": gr.State(None),
    }
    for key in (
        "scan_directory_browse_btn",
        "scan_btn",
        "auto_label_btn",
        "cancel_auto_label_btn",
        "auto_label_output_dir_browse_btn",
        "save_edit_btn",
    ):
        section[key] = gr.Button(key)

    for key in (
        "sample_selector",
        "genre_ratio",
        "auto_label_batch_size",
    ):
        section[key] = gr.Slider(0, 99, value=1)

    for key in (
        "skip_metas",
        "format_lyrics",
        "transcribe_lyrics",
        "only_unlabeled",
        "auto_label_subprocess",
        "all_instrumental",
        "edit_instrumental",
    ):
        section[key] = gr.Checkbox()

    for key in (
        "audio_directory",
        "dataset_name",
        "custom_tag",
        "tag_position",
        "scan_status",
        "lm_lyrics_language",
        "dataset_model_config",
        "dataset_vram_preset",
        "save_path",
        "auto_label_output_dir",
        "label_progress",
        "preview_audio",
        "preview_filename",
        "edit_caption",
        "edit_genre",
        "prompt_override",
        "edit_lyrics",
        "edit_bpm",
        "edit_keyscale",
        "edit_timesig",
        "edit_duration",
        "edit_language",
        "raw_lyrics_display",
        "has_raw_lyrics_state",
        "edit_status",
    ):
        section[key] = gr.Textbox(key)
    return section


if __name__ == "__main__":
    unittest.main()
