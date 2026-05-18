"""Unit tests for subprocess worker console progress."""

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.results.subprocess_worker_progress import (
    WorkerConsoleProgress,
)


class WorkerConsoleProgressTests(unittest.TestCase):
    """Verify worker progress prints useful console output."""

    def test_progress_prints_percent_and_description(self) -> None:
        """Progress callback should print one readable progress line."""

        progress = WorkerConsoleProgress()

        with patch("builtins.print") as print_mock:
            progress(0.52, desc="Preparing inputs...")

        print_mock.assert_called_once_with(
            "[Worker progress] 52% Preparing inputs...",
            flush=True,
        )

    def test_duplicate_progress_line_is_suppressed(self) -> None:
        """Identical progress updates should not spam the console."""

        progress = WorkerConsoleProgress()

        with patch("builtins.print") as print_mock:
            progress(0.8, "Decoding audio...")
            progress(0.8, "Decoding audio...")

        print_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
