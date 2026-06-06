"""Tests for Audio Processing external command progress logging."""

from __future__ import annotations

import sys
import unittest

from acestep.audio_processing.process_logging import run_external_command


class ProcessLoggingTests(unittest.TestCase):
    """Verify streamed subprocess output is parsed into progress callbacks."""

    def test_streamed_percent_output_reaches_callback(self) -> None:
        """Carriage-return percent output should be logged before command exit."""

        messages: list[tuple[float | None, str]] = []
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.write('\\x1b[?25lTask 25.0%\\rTask 50.0%\\n'); "
                "sys.stdout.flush()"
            ),
        ]

        run_external_command(
            cmd,
            "progress probe failed",
            process_callback=lambda progress, label: messages.append((progress, label)),
            timeout=10,
        )

        self.assertIn((0.25, "Task 25.0%"), messages)
        self.assertIn((0.5, "Task 50.0%"), messages)

    def test_ffmpeg_progress_output_uses_duration_percentage(self) -> None:
        """FFmpeg out_time progress should convert to a real percentage."""

        messages: list[tuple[float | None, str]] = []
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.write('out_time_ms=500000\\nprogress=continue\\n"
                "out_time_ms=1000000\\nprogress=end\\n'); "
                "sys.stdout.flush()"
            ),
        ]

        run_external_command(
            cmd,
            "ffmpeg mux failed",
            process_callback=lambda progress, label: messages.append((progress, label)),
            progress_duration_seconds=1.0,
            timeout=10,
        )

        self.assertIn((0.5, "ffmpeg mux: 50.0%"), messages)
        self.assertIn((1.0, "ffmpeg mux: 100.0%"), messages)


if __name__ == "__main__":
    unittest.main()
