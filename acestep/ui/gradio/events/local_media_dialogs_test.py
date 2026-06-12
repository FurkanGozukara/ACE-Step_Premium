"""Tests for local media file picker helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.ui.gradio.events import local_media_dialogs


class _FakeRoot:
    """Minimal Tk root stand-in for media picker tests."""

    def __init__(self) -> None:
        """Track whether the fake dialog root was destroyed."""

        self.destroyed = False

    def destroy(self) -> None:
        """Mark the fake root as destroyed."""

        self.destroyed = True


class LocalMediaDialogsTests(unittest.TestCase):
    """Verify local media picker behavior."""

    def test_select_media_file_path_returns_current_when_dialog_unavailable(self) -> None:
        """Unavailable native dialogs should preserve the current media path."""

        with patch.object(
            local_media_dialogs.local_path_dialogs,
            "is_dialog_available",
            return_value=False,
        ):
            selected = local_media_dialogs.select_media_file_path("C:\\media\\clip.mkv")

        self.assertTrue(selected.lower().endswith("clip.mkv"))

    def test_select_media_file_path_returns_selected_file(self) -> None:
        """Native media picker selections should be normalized and returned."""

        root = _FakeRoot()
        dialog = SimpleNamespace(askopenfilename=lambda **_kwargs: "C:\\media\\clip.mkv")
        with patch.object(
            local_media_dialogs.local_path_dialogs,
            "is_dialog_available",
            return_value=True,
        ):
            with patch.object(
                local_media_dialogs.local_path_dialogs,
                "_create_dialog_root",
                return_value=root,
            ):
                with patch.object(local_media_dialogs.local_path_dialogs, "filedialog", dialog):
                    selected = local_media_dialogs.select_media_file_path("")

        self.assertTrue(selected.lower().endswith("clip.mkv"))
        self.assertTrue(root.destroyed)


if __name__ == "__main__":
    unittest.main()
