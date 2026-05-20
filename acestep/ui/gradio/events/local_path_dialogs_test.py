"""Tests for local file and folder dialog helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events import local_path_dialogs


class _FakeRoot:
    """Minimal Tk root stand-in for dialog helper tests."""

    def __init__(self) -> None:
        """Track whether the fake dialog root was destroyed."""

        self.destroyed = False

    def wm_attributes(self, *_args) -> None:
        """Accept topmost window attributes."""

    def withdraw(self) -> None:
        """Accept hiding the fake root window."""

    def destroy(self) -> None:
        """Mark the fake root as destroyed."""

        self.destroyed = True


class LocalPathDialogsTests(unittest.TestCase):
    """Verify native picker helpers normalize paths and handle cancel states."""

    def setUp(self) -> None:
        """Preserve configured safe roots."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore configured safe roots."""

        set_safe_roots(self._safe_roots)

    def test_normalize_dialog_path_strips_quotes(self) -> None:
        """Quoted user paths should be normalized before use."""

        path = local_path_dialogs.normalize_dialog_path('"C:\\temp\\dataset.json"')

        self.assertTrue(path.lower().endswith(os.path.join("temp", "dataset.json")))

    def test_select_folder_path_returns_current_when_dialog_unavailable(self) -> None:
        """Headless or unavailable dialogs should not erase the current path."""

        with patch.object(local_path_dialogs, "is_dialog_available", return_value=False):
            selected = local_path_dialogs.select_folder_path("C:\\temp")

        self.assertTrue(selected.lower().endswith("temp"))

    def test_select_optional_folder_path_returns_none_on_cancel(self) -> None:
        """Optional folder selection should distinguish cancel from selection."""

        root = _FakeRoot()
        dialog = SimpleNamespace(askdirectory=lambda **_kwargs: "")
        with patch.object(local_path_dialogs, "is_dialog_available", return_value=True):
            with patch.object(local_path_dialogs, "Tk", return_value=root):
                with patch.object(local_path_dialogs, "filedialog", dialog):
                    selected = local_path_dialogs.select_optional_folder_path("C:\\temp")

        self.assertIsNone(selected)
        self.assertTrue(root.destroyed)

    def test_select_optional_folder_path_allows_selecting_current_path(self) -> None:
        """Selecting the current folder should still count as a selection."""

        root = _FakeRoot()
        dialog = SimpleNamespace(askdirectory=lambda **_kwargs: "C:\\temp")
        with patch.object(local_path_dialogs, "is_dialog_available", return_value=True):
            with patch.object(local_path_dialogs, "Tk", return_value=root):
                with patch.object(local_path_dialogs, "filedialog", dialog):
                    selected = local_path_dialogs.select_optional_folder_path("C:\\temp")

        self.assertTrue(selected.lower().endswith("temp"))
        self.assertTrue(root.destroyed)

    def test_select_optional_json_file_path_returns_none_on_cancel(self) -> None:
        """Optional JSON selection should distinguish cancel from selection."""

        root = _FakeRoot()
        dialog = SimpleNamespace(askopenfilename=lambda **_kwargs: "")
        with patch.object(local_path_dialogs, "is_dialog_available", return_value=True):
            with patch.object(local_path_dialogs, "Tk", return_value=root):
                with patch.object(local_path_dialogs, "filedialog", dialog):
                    selected = local_path_dialogs.select_optional_json_file_path(
                        "C:\\temp\\dataset.json"
                    )

        self.assertIsNone(selected)
        self.assertTrue(root.destroyed)

    def test_select_optional_json_file_path_allows_selecting_current_file(self) -> None:
        """Selecting the current JSON file should still count as a selection."""

        root = _FakeRoot()
        dialog = SimpleNamespace(
            askopenfilename=lambda **_kwargs: "C:\\temp\\dataset.json"
        )
        with patch.object(local_path_dialogs, "is_dialog_available", return_value=True):
            with patch.object(local_path_dialogs, "Tk", return_value=root):
                with patch.object(local_path_dialogs, "filedialog", dialog):
                    selected = local_path_dialogs.select_optional_json_file_path(
                        "C:\\temp\\dataset.json"
                    )

        self.assertTrue(selected.lower().endswith(os.path.join("temp", "dataset.json")))
        self.assertTrue(root.destroyed)

    def test_select_json_save_path_appends_json_extension(self) -> None:
        """Save dialogs should enforce the expected JSON extension."""

        root = _FakeRoot()
        dialog = SimpleNamespace(asksaveasfilename=lambda **_kwargs: "C:\\temp\\dataset")
        with patch.object(local_path_dialogs, "is_dialog_available", return_value=True):
            with patch.object(local_path_dialogs, "Tk", return_value=root):
                with patch.object(local_path_dialogs, "filedialog", dialog):
                    selected = local_path_dialogs.select_json_save_path("")

        self.assertTrue(selected.lower().endswith(os.path.join("temp", "dataset.json")))
        self.assertTrue(root.destroyed)

    def test_select_pt_file_path_opens_state_file_picker(self) -> None:
        """Resume state picker should select .pt files."""

        root = _FakeRoot()
        dialog = SimpleNamespace(
            askopenfilename=lambda **_kwargs: "C:\\temp\\epoch-3-training_resume_state.pt"
        )
        with patch.object(local_path_dialogs, "is_dialog_available", return_value=True):
            with patch.object(local_path_dialogs, "Tk", return_value=root):
                with patch.object(local_path_dialogs, "filedialog", dialog):
                    selected = local_path_dialogs.select_pt_file_path("")

        self.assertTrue(selected.lower().endswith("epoch-3-training_resume_state.pt"))
        self.assertTrue(root.destroyed)

    def test_open_folder_path_uses_windows_file_explorer(self) -> None:
        """Folder opener should create and open the requested folder on Windows."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            target = os.path.join(tmpdir, "Loras")
            target_real = os.path.realpath(target)
            with patch.object(local_path_dialogs.sys, "platform", "win32"):
                with patch.object(
                    local_path_dialogs.os,
                    "startfile",
                    create=True,
                ) as startfile:
                    status = local_path_dialogs.open_folder_path(target)
                    created = os.path.isdir(target)

        self.assertIn("Opened folder", status)
        startfile.assert_called_once_with(target_real)
        self.assertTrue(created)

    def test_open_folder_path_uses_linux_file_manager(self) -> None:
        """Folder opener should use a common Linux opener when available."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            target = os.path.join(tmpdir, "Loras")
            target_real = os.path.realpath(target)
            with patch.object(local_path_dialogs.sys, "platform", "linux"):
                with patch.object(
                    local_path_dialogs.shutil,
                    "which",
                    side_effect=lambda command: "/usr/bin/xdg-open"
                    if command == "xdg-open"
                    else None,
                ):
                    with patch.object(local_path_dialogs.subprocess, "Popen") as popen:
                        status = local_path_dialogs.open_folder_path(target)
                        created = os.path.isdir(target)

        self.assertIn("Opened folder", status)
        popen.assert_called_once_with(["xdg-open", target_real])
        self.assertTrue(created)


if __name__ == "__main__":
    unittest.main()
