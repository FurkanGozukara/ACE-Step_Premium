"""Tests for DatasetHandler import behavior."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from acestep.dataset_handler import DatasetHandler
from acestep.training.dataset_builder_modules.models import AudioSample


class DatasetHandlerTests(unittest.TestCase):
    """Verify Dataset Explorer import accepts JSON files and audio folders."""

    def test_import_dataset_requires_path(self) -> None:
        """Import should explain that a JSON or folder path is required."""

        handler = DatasetHandler()

        status = handler.import_dataset("train", "")

        self.assertIn("dataset JSON", status)
        self.assertFalse(handler.dataset_imported)

    def test_import_dataset_loads_json_file(self) -> None:
        """JSON paths should be loaded through DatasetBuilder.load_dataset."""

        with patch("acestep.dataset_handler.DatasetBuilder") as builder_cls:
            builder = builder_cls.return_value
            builder.load_dataset.return_value = ([object()], "Loaded")
            builder.get_labeled_count.return_value = 1

            handler = DatasetHandler()
            status = handler.import_dataset("train", "C:\\temp\\dataset.json")

        builder.load_dataset.assert_called_once_with("C:\\temp\\dataset.json")
        builder.scan_directory.assert_not_called()
        self.assertTrue(handler.dataset_imported)
        self.assertIn("Samples: 1 (1 labeled)", status)

    def test_import_dataset_loads_quoted_json_path_with_spaces(self) -> None:
        """Quoted Dataset page paths should be classified after normalization."""

        with patch("acestep.dataset_handler.DatasetBuilder") as builder_cls:
            builder = builder_cls.return_value
            builder.load_dataset.return_value = ([object()], "Loaded")
            builder.get_labeled_count.return_value = 1

            handler = DatasetHandler()
            status = handler.import_dataset("train", '"./datasets/my data.json"')

        builder.load_dataset.assert_called_once_with(
            os.path.normpath("./datasets/my data.json")
        )
        builder.scan_directory.assert_not_called()
        self.assertTrue(handler.dataset_imported)
        self.assertIn("Samples: 1 (1 labeled)", status)

    def test_import_dataset_scans_audio_folder(self) -> None:
        """Non-JSON paths should be treated as audio folders."""

        with patch("acestep.dataset_handler.DatasetBuilder") as builder_cls:
            builder = builder_cls.return_value
            builder.scan_directory.return_value = ([object(), object()], "Scanned")
            builder.get_labeled_count.return_value = 0

            handler = DatasetHandler()
            status = handler.import_dataset("test", "C:\\temp\\audio")

        builder.scan_directory.assert_called_once_with("C:\\temp\\audio")
        builder.load_dataset.assert_not_called()
        self.assertEqual("test_dataset", builder.metadata.name)
        self.assertTrue(handler.dataset_imported)
        self.assertIn("Samples: 2 (0 labeled)", status)

    def test_import_dataset_for_ui_returns_first_sample_preview(self) -> None:
        """Dataset page imports should immediately show the first sample."""

        sample = AudioSample(
            audio_path="C:\\temp\\song.mp3",
            filename="song.mp3",
            caption="A bright song",
            labeled=True,
        )
        with patch("acestep.dataset_handler.DatasetBuilder") as builder_cls:
            builder = builder_cls.return_value
            builder.load_dataset.return_value = ([sample], "Loaded")
            builder.get_labeled_count.return_value = 1

            handler = DatasetHandler()
            status, instruction, metadata, src_audio, target_audio, ref_audio = (
                handler.import_dataset_for_ui("train", "C:\\temp\\dataset.json")
            )

        self.assertIn("Samples: 1 (1 labeled)", status)
        self.assertEqual("A bright song", instruction)
        self.assertIn('"filename": "song.mp3"', metadata)
        self.assertEqual("C:\\temp\\song.mp3", src_audio)
        self.assertIsNone(target_audio)
        self.assertIsNone(ref_audio)

    def test_get_item_for_ui_resolves_index(self) -> None:
        """Get Item should preview an imported sample by index."""

        handler = DatasetHandler()
        handler.dataset = [
            AudioSample(filename="first.mp3", audio_path="C:\\temp\\first.mp3"),
            AudioSample(filename="second.mp3", audio_path="C:\\temp\\second.mp3"),
        ]
        handler.dataset_imported = True

        status, _instruction, metadata, src_audio, _target_audio, _ref_audio = (
            handler.get_item_for_ui("idx", "1")
        )

        self.assertIn("Loaded item 2/2", status)
        self.assertIn('"filename": "second.mp3"', metadata)
        self.assertEqual("C:\\temp\\second.mp3", src_audio)


if __name__ == "__main__":
    unittest.main()
