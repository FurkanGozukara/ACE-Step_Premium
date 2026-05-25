"""Tests for Dataset page import output formatting."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.wiring.dataset_import_outputs import finish_dataset_import


class DatasetImportOutputsTests(unittest.TestCase):
    """Verify Dataset import output contracts used by Gradio wiring."""

    def test_ngram_choices_display_song_count_without_hit_count(self) -> None:
        """N-gram selector choices should show only the gram and song count."""

        class ImportedDataset:
            """Minimal handler state for finish_dataset_import."""

            dataset_imported = True

        result = (
            "Loaded",
            "caption",
            "{}",
            None,
            None,
            None,
            [["red blue", 2, 9]],
            [],
            [],
            [],
            [],
            [],
            "Select a gram from the columns above.",
            [],
            "",
            0,
        )

        outputs = finish_dataset_import(ImportedDataset(), result)

        self.assertEqual(("red blue  |  2 songs", "red blue"), outputs[7]["choices"][0])
        self.assertNotIn("hits", outputs[7]["choices"][0][0])


if __name__ == "__main__":
    unittest.main()
