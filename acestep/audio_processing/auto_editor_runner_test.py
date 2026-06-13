"""Tests for Auto-Editor trim command construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.audio_processing.auto_editor_runner import run_auto_editor
from acestep.audio_processing.auto_editor_trim_settings import AutoEditorTrimSettings


class AutoEditorRunnerTests(unittest.TestCase):
    """Verify command arguments passed to the bundled Auto-Editor binary."""

    @patch("acestep.audio_processing.auto_editor_runner.run_command")
    @patch("acestep.audio_processing.auto_editor_runner.auto_editor_command")
    def test_zero_db_threshold_reaches_auto_editor_command(
        self,
        command_mock,
        run_mock,
    ) -> None:
        """A 0 dB trim threshold should be forwarded without fallback coercion."""

        command_mock.return_value = ["auto-editor"]

        with tempfile.TemporaryDirectory() as temp_dir:
            run_auto_editor(
                Path(temp_dir) / "analysis.wav",
                Path(temp_dir) / "timeline.v3",
                AutoEditorTrimSettings(threshold_db=0.0),
            )

        cmd = run_mock.call_args.args[0]
        self.assertIn("--edit", cmd)
        self.assertIn("audio:threshold=0dB", cmd)


if __name__ == "__main__":
    unittest.main()
