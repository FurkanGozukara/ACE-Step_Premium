"""Tests for Auto-Editor workflow export helpers."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from acestep.audio_processing.auto_editor_trim_settings import AutoEditorTrimSettings
from acestep.audio_processing.auto_editor_workflow import (
    AUTO_EDITOR_WORKFLOW_EXPORT_NONE,
    export_auto_editor_workflow,
    normalize_workflow_export_mode,
)


class AutoEditorWorkflowExportTests(unittest.TestCase):
    """Verify workflow-only Auto-Editor exports."""

    @patch("acestep.audio_processing.auto_editor_workflow.run_command")
    @patch("acestep.audio_processing.auto_editor_workflow.auto_editor_command")
    def test_exports_resolve_workflow_without_media_render_args(
        self,
        command_mock,
        run_mock,
    ) -> None:
        """Workflow export should call Auto-Editor with only timeline export args."""

        command_mock.return_value = ["auto-editor"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = export_auto_editor_workflow(
                "input.mp4",
                temp_dir,
                "clip",
                "resolve",
                AutoEditorTrimSettings(threshold_db=-40.0, margin_seconds=0.1),
            )

        cmd = run_mock.call_args.args[0]
        self.assertTrue(output.endswith("clip.fcpxml"))
        self.assertIn("--export", cmd)
        self.assertIn("resolve", cmd)
        self.assertIn("-o", cmd)
        self.assertNotIn("--video-codec", cmd)
        self.assertNotIn("--audio-codec", cmd)
        self.assertIn("--progress", cmd)
        self.assertIn("ascii", cmd)

    @patch("acestep.audio_processing.auto_editor_workflow.run_command")
    @patch("acestep.audio_processing.auto_editor_workflow.auto_editor_command")
    def test_resolve_workflow_rewrites_fcpxml_media_reference(
        self,
        command_mock,
        _run_mock,
    ) -> None:
        """FCPXML export should point to the intended media file, not a temp upload."""

        command_mock.return_value = ["auto-editor"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staged_media = root / "gradio" / "0.mkv"
            intended_media = root / "whisper tutorial" / "0.mkv"
            staged_media.parent.mkdir()
            intended_media.parent.mkdir()
            staged_media.touch()
            intended_media.touch()
            workflow = root / "clip.fcpxml"
            workflow.write_text(
                (
                    "<?xml version='1.0' encoding='utf-8'?>\n"
                    "<fcpxml><resources><asset id=\"r2\">"
                    f"<media-rep src=\"{staged_media.resolve().as_uri()}\" "
                    "kind=\"original-media\" />"
                    "</asset></resources></fcpxml>"
                ),
                encoding="utf-8",
            )

            export_auto_editor_workflow(
                staged_media,
                root,
                "clip",
                "resolve",
                AutoEditorTrimSettings(threshold_db=-40.0, margin_seconds=0.1),
                media_reference=intended_media,
            )

            media_rep = ET.parse(workflow).getroot().find(".//media-rep")
            self.assertIsNotNone(media_rep)
            self.assertEqual(intended_media.resolve().as_uri(), media_rep.get("src"))

    def test_invalid_workflow_mode_normalizes_to_none(self) -> None:
        """Unsupported saved workflow modes should fall back to None."""

        self.assertEqual(AUTO_EDITOR_WORKFLOW_EXPORT_NONE, normalize_workflow_export_mode("bad"))


if __name__ == "__main__":
    unittest.main()
