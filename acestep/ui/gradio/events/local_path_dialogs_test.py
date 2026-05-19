"""Tests for local file and folder dialog helpers."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_normalize_dialog_path_strips_quotes(self) -> None:
        """Quoted user paths should be normalized before use."""

        path = local_path_dialogs.normalize_dialog_path('"C:\\temp\\dataset.json"')

        self.assertTrue(path.lower().endswith(os.path.join("temp", "dataset.json")))

    def test_select_folder_path_returns_current_when_dialog_unavailable(self) -> None:
        """Headless or unavailable dialogs should not erase the current path."""

        with patch.object(local_path_dialogs, "is_dialog_available", return_value=False):
            selected = local_path_dialogs.select_folder_path("C:\\temp")

        self.assertTrue(selected.lower().endswith("temp"))

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


if __name__ == "__main__":
    unittest.main()
