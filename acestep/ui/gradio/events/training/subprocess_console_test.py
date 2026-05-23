"""Tests for training subprocess console output helpers."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.training.subprocess_console import write_console_text


class _StrictCp1252Stream:
    """Stream that raises like a cp1252 Windows console for unsupported text."""

    encoding = "cp1252"

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> None:
        """Capture text only when cp1252 can encode it."""

        text.encode(self.encoding)
        self.chunks.append(text)

    def flush(self) -> None:
        """No-op flush for stream compatibility."""


class SubprocessConsoleTests(unittest.TestCase):
    """Verify Unicode status text cannot crash console mirroring."""

    def test_write_console_text_replaces_unsupported_windows_console_chars(self) -> None:
        """Emoji status text should be mirrored safely to cp1252 streams."""

        stream = _StrictCp1252Stream()

        write_console_text("🚀 Starting training", end="\n", stream=stream)

        self.assertEqual("? Starting training\n", "".join(stream.chunks))


if __name__ == "__main__":
    unittest.main()
