"""Tests for DatasetHandler import behavior."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
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
            result = handler.import_dataset_for_ui("train", "C:\\temp\\dataset.json")
            status, instruction, metadata, src_audio, target_audio, ref_audio = result[:6]

        self.assertIn("Samples: 1 (1 labeled)", status)
        self.assertEqual("A bright song", instruction)
        self.assertIn('"filename": "song.mp3"', metadata)
        self.assertEqual("C:\\temp\\song.mp3", src_audio)
        self.assertIsNone(target_audio)
        self.assertIsNone(ref_audio)

    def test_import_dataset_for_ui_returns_top_word_ngram_tables(self) -> None:
        """Dataset page imports should populate 1-gram through 6-gram tables."""

        samples = [
            AudioSample(
                filename="first.wav",
                audio_path="C:\\temp\\first.wav",
                caption="red blue red blue",
                genre="funk style",
                lyrics="lyriconly lyriconly lyriconly",
                labeled=True,
            ),
            AudioSample(
                filename="second.wav",
                audio_path="C:\\temp\\second.wav",
                caption="red blue green yellow",
                genre="soul style",
                lyrics="lyriconly hiddenwords",
                labeled=True,
            ),
        ]
        with patch("acestep.dataset_handler.DatasetBuilder") as builder_cls:
            builder = builder_cls.return_value
            builder.load_dataset.return_value = (samples, "Loaded")
            builder.get_labeled_count.return_value = 2

            handler = DatasetHandler()
            result = handler.import_dataset_for_ui("train", "C:\\temp\\dataset.json")

        one_grams, two_grams, three_grams, four_grams, five_grams, six_grams = (
            result[6:12]
        )
        self.assertIn(["red", 2, 3], one_grams[:2])
        self.assertEqual(["red blue", 2, 3], two_grams[0])
        self.assertIn(["red blue green", 1, 1], three_grams)
        self.assertIn(["red blue green yellow", 1, 1], four_grams)
        self.assertIn(["red blue green yellow soul", 1, 1], five_grams)
        self.assertIn(["red blue green yellow soul style", 1, 1], six_grams)
        self.assertNotIn("lyriconly", [row[0] for row in one_grams])

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

    def test_select_ngram_lists_songs_and_loads_clicked_song(self) -> None:
        """Selecting a gram should expose matching songs that preview on click."""

        handler = DatasetHandler()
        handler.dataset = [
            AudioSample(
                filename="first.wav",
                audio_path="C:\\temp\\first.wav",
                caption="First song",
                lyrics="red blue red blue",
                genre="west coast west coast rap",
            ),
            AudioSample(
                filename="second.wav",
                audio_path="C:\\temp\\second.wav",
                caption="Second song",
                lyrics="red blue green",
                genre="west coast west coast soul",
            ),
        ]
        handler.dataset_imported = True

        summary, rows, selected_gram, selected_size = handler.select_ngram_for_ui(
            2,
            SimpleNamespace(index=[0, 0]),
        )

        self.assertIn("2-gram: west coast", summary)
        self.assertEqual("west coast", selected_gram)
        self.assertEqual(2, selected_size)
        self.assertEqual(["first", "second"], [row[1] for row in rows])

        status, instruction, metadata, src_audio, _target_audio, _ref_audio = (
            handler.select_ngram_song_for_ui(
                selected_gram,
                selected_size,
                SimpleNamespace(index=[1, 0]),
            )
        )

        self.assertIn("Loaded song 2/2: second", status)
        self.assertEqual("Second song", instruction)
        self.assertIn('"genre": "west coast west coast soul"', metadata)
        self.assertEqual("C:\\temp\\second.wav", src_audio)


if __name__ == "__main__":
    unittest.main()
