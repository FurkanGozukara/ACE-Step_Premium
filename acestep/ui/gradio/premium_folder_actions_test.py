"""Tests for cross-platform folder-opening actions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acestep.ui.gradio import premium_features


class PremiumFolderActionsTests(unittest.TestCase):
    """Verify folder actions use robust platform launch behavior."""

    def test_windows_uses_startfile_for_existing_folder(self) -> None:
        """Windows folder opening should prefer the native shell API."""

        with tempfile.TemporaryDirectory() as tmpdir:
            startfile = MagicMock()
            with patch.object(premium_features.sys, "platform", "win32"):
                with patch.object(premium_features.os, "startfile", startfile, create=True):
                    status = premium_features.open_folder_in_system(tmpdir)

        self.assertIn("Opened", status)
        startfile.assert_called_once()
        self.assertEqual(Path(tmpdir).resolve(), Path(startfile.call_args.args[0]))

    def test_windows_falls_back_to_explorer_when_startfile_is_missing(self) -> None:
        """Windows folder opening should still work without ``os.startfile``."""

        with tempfile.TemporaryDirectory() as tmpdir:
            popen = MagicMock()
            with patch.object(premium_features.sys, "platform", "win32"):
                with patch.object(premium_features.os, "startfile", None, create=True):
                    with patch.object(
                        premium_features.shutil,
                        "which",
                        return_value="explorer.exe",
                    ):
                        with patch.object(premium_features.subprocess, "Popen", popen):
                            status = premium_features.open_folder_in_system(tmpdir)

        self.assertIn("Opened", status)
        popen.assert_called_once()
        self.assertEqual("explorer.exe", popen.call_args.args[0][0])
        self.assertEqual(Path(tmpdir).resolve(), Path(popen.call_args.args[0][1]))

    def test_linux_uses_first_available_file_explorer(self) -> None:
        """Linux folder opening should not assume only ``xdg-open`` exists."""

        def which(command: str) -> str | None:
            return "/usr/bin/gio" if command == "gio" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            popen = MagicMock()
            with patch.object(premium_features.sys, "platform", "linux"):
                with patch.object(premium_features.shutil, "which", side_effect=which):
                    with patch.object(premium_features.subprocess, "Popen", popen):
                        status = premium_features.open_folder_in_system(tmpdir)

        self.assertIn("Opened", status)
        popen.assert_called_once()
        self.assertEqual(
            ["/usr/bin/gio", "open", str(Path(tmpdir).resolve())],
            popen.call_args.args[0],
        )

    def test_missing_linux_file_explorer_reports_folder_path(self) -> None:
        """Headless Linux should return the folder path instead of raising."""

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(premium_features.sys, "platform", "linux"):
                with patch.object(premium_features.shutil, "which", return_value=None):
                    status = premium_features.open_folder_in_system(tmpdir)

        self.assertIn("Folder is available", status)
        self.assertIn(str(Path(tmpdir).resolve()), status)

    def test_file_path_opens_parent_folder(self) -> None:
        """Passing a file path should open the containing directory."""

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "song.mp3"
            file_path.write_bytes(b"")
            startfile = MagicMock()
            with patch.object(premium_features.sys, "platform", "win32"):
                with patch.object(premium_features.os, "startfile", startfile, create=True):
                    premium_features.open_folder_in_system(file_path)

        self.assertEqual(Path(tmpdir).resolve(), Path(startfile.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
