"""Tests for Audio Processing tab event helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from acestep.audio_processing.settings import UI_SETTING_KEYS
from acestep.ui.gradio.events.wiring.audio_processing_process_status import (
    make_process_log_callback,
    open_audio_processing_outputs_folder,
)
from acestep.ui.gradio.events.wiring.audio_processing_wiring import (
    _preview_diffpitcher_reference,
    _preview_single_file,
    _process_single_file,
    _toggle_audio_enhancement_stages,
)
from acestep.ui.gradio.pages.audio_processing_page import create_audio_processing_page


class AudioProcessingWiringTests(unittest.TestCase):
    """Verify Audio Processing UI helper behavior."""

    def test_toggle_stage_checkboxes_unchecks_when_all_enabled(self) -> None:
        """The toggle should uncheck all enabled processing stages."""

        updates = _toggle_audio_enhancement_stages(*([True] * 12))

        self.assertEqual(12, len(updates))
        self.assertTrue(updates)
        self.assertTrue(all(update["value"] is False for update in updates))

    def test_toggle_stage_checkboxes_checks_when_any_disabled(self) -> None:
        """The toggle should check all stages when any stage is off."""

        updates = _toggle_audio_enhancement_stages(*([True] * 6), False, *([True] * 5))

        self.assertTrue(all(update["value"] is True for update in updates))

    def test_process_file_workflow_export_skips_processed_media(self) -> None:
        """Selected workflow export should return only the Auto-Editor workflow file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_path = f"{temp_dir}/clip.fcpxml"
            with (
                patch(
                    "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                    "create_audio_processing_run_dir",
                    return_value=temp_dir,
                ),
                patch(
                    "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                    "export_auto_editor_workflow",
                    return_value=workflow_path,
                ) as export_workflow,
                patch(
                    "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                    "process_media_file",
                ) as process_media,
            ):
                audio, video, figure, files, status = _process_single_file(
                    "clip.mp4",
                    None,
                    None,
                    *_settings_values(workflow_export="resolve"),
                )

        export_workflow.assert_called_once()
        process_media.assert_not_called()
        self.assertFalse(audio["visible"])
        self.assertFalse(figure["visible"])
        self.assertFalse(video["visible"])
        self.assertEqual([workflow_path], files["value"])
        self.assertTrue(files["visible"])
        self.assertIn("Export complete", status)
        self.assertNotIn("### Process Log", status)
        self.assertIn("Processed audio/video: `None`", status)

    def test_workflow_export_allows_temp_upload_with_reference_note(self) -> None:
        """Workflow export should still run when only a Gradio temp upload is available."""

        source = Path(tempfile.gettempdir()) / "gradio" / "upload" / "0.mkv"
        with (
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "create_audio_processing_run_dir",
                return_value="run",
            ),
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "export_auto_editor_workflow",
                return_value="run/0.fcpxml",
            ) as export_workflow,
        ):
            audio, video, figure, files, status = _process_single_file(
                str(source),
                None,
                None,
                *_settings_values(workflow_export="resolve"),
            )

        export_workflow.assert_called_once()
        self.assertFalse(audio["visible"])
        self.assertFalse(figure["visible"])
        self.assertFalse(video["visible"])
        self.assertEqual(["run/0.fcpxml"], files["value"])
        self.assertTrue(files["visible"])
        self.assertIn("Gradio temp upload", status)

    def test_workflow_export_prefers_explicit_local_media_path(self) -> None:
        """Workflow export should use a local path instead of a temp upload when provided."""

        with (
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "create_audio_processing_run_dir",
                return_value="run",
            ),
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "export_auto_editor_workflow",
                return_value="run/clip.fcpxml",
            ) as export_workflow,
        ):
            _process_single_file(
                "C:/Users/Furkan/AppData/Local/Temp/gradio/upload/0.mkv",
                None,
                '"C:\\whisper tutorial\\0.mkv"',
                *_settings_values(workflow_export="resolve"),
            )

        self.assertEqual(
            "C:/Users/Furkan/AppData/Local/Temp/gradio/upload/0.mkv",
            export_workflow.call_args.args[0],
        )
        self.assertEqual(
            "C:\\whisper tutorial\\0.mkv",
            export_workflow.call_args.kwargs["media_reference"],
        )

    def test_process_file_appends_process_log_to_status(self) -> None:
        """External process callback messages should appear in the status markdown."""

        result = SimpleNamespace(
            audio_path="processed.wav",
            video_path=None,
            file_list=lambda: ["processed.wav", "metadata.json"],
            processed_audio=SimpleNamespace(before=None, after=None, sample_rate=48000),
        )

        def _process_side_effect(*_args, **kwargs):
            kwargs["progress_callback"](None, "Auto-Editor process started")
            kwargs["progress_callback"](None, "Auto-Editor reencode finished")
            return result

        with (
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "create_audio_processing_run_dir",
                return_value="run",
            ),
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "process_media_file",
                side_effect=_process_side_effect,
            ),
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "make_spectrogram_figure",
                return_value="figure",
            ),
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
                "metrics_markdown",
                return_value="metrics",
            ),
        ):
            _audio, _video, _figure, _files, status = _process_single_file(
                "clip.mp4",
                None,
                None,
                *_settings_values(),
            )

        self.assertIn("### Process Log", status)
        self.assertIn("Auto-Editor process started", status)
        self.assertIn("Auto-Editor reencode finished", status)

    def test_process_log_callback_updates_gradio_progress(self) -> None:
        """Numeric progress values should update the Gradio progress hook."""

        process_log: list[str] = []
        progress_calls: list[tuple[float, str]] = []

        def _progress(value: float, desc: str) -> None:
            progress_calls.append((value, desc))

        callback = make_process_log_callback(process_log, _progress)

        callback(0.5, "External command running")

        self.assertEqual([(0.5, "External command running")], progress_calls)
        self.assertEqual(["50.0% - External command running"], process_log)

    def test_preview_file_workflow_export_skips_preview_audio(self) -> None:
        """Selected workflow export should not save preview audio."""

        with patch(
            "acestep.ui.gradio.events.wiring.audio_processing_single_file_handlers."
            "process_media_file"
        ) as process_media:
            before, after, figure, files, status = _preview_single_file(
                "clip.mp4",
                None,
                None,
                *_settings_values(workflow_export="resolve"),
            )

        process_media.assert_not_called()
        self.assertIsNone(before)
        self.assertIsNone(after)
        self.assertIsNone(figure)
        self.assertFalse(files["visible"])
        self.assertIn("Preview skipped", status)

    @patch(
        "acestep.ui.gradio.events.wiring.audio_processing_process_status."
        "open_folder_path"
    )
    def test_open_outputs_folder_uses_default_outputs_root(self, open_folder_mock) -> None:
        """The outputs-folder button should open the default Audio Processing root."""

        open_folder_mock.return_value = "Opened folder"

        status = open_audio_processing_outputs_folder()

        self.assertEqual("Opened folder", status)
        self.assertTrue(open_folder_mock.call_args.args[0].endswith("outputs"))

    def test_audio_processing_page_exposes_all_generation_settings(self) -> None:
        """Every ordered Audio Processing setting should have a concrete UI control."""

        with gr.Blocks():
            controls = create_audio_processing_page()

        self.assertIn("ap_disable_upload_preview", controls)
        self.assertIn("ap_single_local_path", controls)
        self.assertIn("ap_single_local_path_browse_btn", controls)
        for key in UI_SETTING_KEYS:
            self.assertIn(key, controls)

    def test_diffpitcher_reference_preview_supports_video_files(self) -> None:
        """DiffPitcher reference uploads should preview video references."""

        audio_update, video_update, status = _preview_diffpitcher_reference(
            {"path": "C:/music/reference.mp4"}
        )

        self.assertFalse(audio_update["visible"])
        self.assertTrue(video_update["visible"])
        self.assertEqual("C:/music/reference.mp4", video_update["value"])
        self.assertIn("Loaded reference video", status)


def _settings_values(workflow_export: str = "none") -> list[object]:
    """Return Audio Processing UI values for wiring-helper tests."""

    values: list[object] = []
    for key in UI_SETTING_KEYS:
        if key == "ap_auto_editor_workflow_export":
            values.append(workflow_export)
        elif key.endswith("_enabled"):
            values.append(True)
        elif key == "ap_output_format":
            values.append("wav")
        else:
            values.append(None)
    return values


if __name__ == "__main__":
    unittest.main()
