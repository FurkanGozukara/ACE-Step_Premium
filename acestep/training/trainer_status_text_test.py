"""Regression tests for trainer status text shown in Gradio."""

from __future__ import annotations

import unittest
from pathlib import Path


class TrainerStatusTextTests(unittest.TestCase):
    """Verify trainer status strings are stored as valid UTF-8 text."""

    def test_trainer_status_text_has_no_mojibake_markers(self) -> None:
        """Trainer UI statuses should not contain broken UTF-8 character sequences."""

        source = (Path(__file__).resolve().parent / "trainer.py").read_text(
            encoding="utf-8"
        )

        markers = (
            "\u00c3",
            "\u00f0\u0178",
            "\u00e2\u0161",
            "\u00e2\u009d",
            "\u00e2\u201e",
            "\u00e2\u008f",
            "\u00e2\u0153",
        )
        for marker in markers:
            self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
