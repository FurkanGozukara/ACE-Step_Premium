"""Tests for Dataset page n-gram event wiring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from acestep.training.dataset_builder_modules.models import AudioSample
from acestep.ui.gradio.events.wiring.dataset_import_wiring import (
    _make_ngram_value_handler,
)


class DatasetImportWiringTests(unittest.TestCase):
    """Verify Dataset n-gram callbacks return UI-friendly choices."""

    def test_matching_song_choices_hide_hit_counts(self) -> None:
        """Matching-song labels should show song, duration, and style only."""

        handler = SimpleNamespace(
            dataset=[
                AudioSample(
                    filename="song.wav",
                    audio_path="C:\\temp\\song.wav",
                    caption="red blue red blue",
                    genre="west coast",
                    duration=12,
                )
            ]
        )
        select_ngram = _make_ngram_value_handler(handler, 2)

        _summary, song_update, _gram, _size = select_ngram("red blue")

        self.assertEqual(
            [("song  |  12.0s  |  west coast", "0")],
            song_update["choices"],
        )
        self.assertNotIn("hits", song_update["choices"][0][0])


if __name__ == "__main__":
    unittest.main()
