"""Tests for batch folder input scanning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.batch_folder_files import (
    discover_batch_folder_items,
    resolve_existing_input_folder,
    resolve_output_folder,
)
from acestep.ui.gradio.events.results.output_manager import use_results_dir


class BatchFolderFilesTests(unittest.TestCase):
    """Verify lyrics/style text-file discovery."""

    def test_discovers_lyrics_with_optional_style_companion(self):
        """Lyrics files should pair with matching ``*_style.txt`` files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song_b.txt").write_text("[verse]\nB", encoding="utf-8")
            (root / "song_b_style.txt").write_text("rock style", encoding="utf-8")
            (root / "song_a.txt").write_text("[verse]\nA", encoding="utf-8")

            items = discover_batch_folder_items(root)

            self.assertEqual(["song_a", "song_b"], [item.stem for item in items])
            self.assertEqual("", items[0].style)
            self.assertEqual("rock style", items[1].style)
            self.assertEqual(root / "song_b_style.txt", items[1].style_path)

    def test_style_only_files_are_not_generated(self):
        """A style companion without lyrics should not become a batch item."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "lonely_style.txt").write_text("style only", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No lyrics"):
                discover_batch_folder_items(root)

    def test_empty_output_folder_uses_default_results_dir(self):
        """Leaving the output folder empty should use the active default outputs."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "outputs"
            with use_results_dir(target):
                output = resolve_output_folder("")

            self.assertEqual(target.resolve(), output)
            self.assertTrue(output.is_dir())

    def test_quoted_input_folder_with_spaces_resolves(self):
        """Batch folder paths pasted with quotes and spaces should resolve."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "batch inputs"
            root.mkdir()

            resolved = resolve_existing_input_folder(f'"{root}"')

        self.assertEqual(root.resolve(), resolved)

    def test_quoted_output_folder_with_spaces_resolves(self):
        """Output folder paths pasted with quotes and spaces should resolve."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "batch outputs"

            resolved = resolve_output_folder(f'"{root}"')

            self.assertEqual(root.resolve(), resolved)
            self.assertTrue(resolved.is_dir())


if __name__ == "__main__":
    unittest.main()
