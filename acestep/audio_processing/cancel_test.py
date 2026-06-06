"""Tests for Audio Processing subprocess cancellation."""

from __future__ import annotations

import unittest

from acestep.audio_processing.cancel import (
    register_audio_processing_subprocess,
    request_audio_processing_cancel,
    unregister_audio_processing_subprocess,
)


class AudioProcessingCancelTests(unittest.TestCase):
    """Verify Audio Processing subprocess cancellation state."""

    def test_registered_subprocess_is_terminated(self) -> None:
        """Registered subprocesses should be terminated on cancel."""

        process = _FakeProcess()
        register_audio_processing_subprocess(process)
        try:
            self.assertTrue(request_audio_processing_cancel())
            self.assertTrue(process.terminated)
        finally:
            unregister_audio_processing_subprocess(process)

    def test_cancel_without_subprocess_reports_no_work(self) -> None:
        """Cancel should report no work when no subprocess is active."""

        self.assertFalse(request_audio_processing_cancel())


class _FakeProcess:
    """Minimal process object for cancellation tests."""

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        """Return active status until terminated."""

        return 1 if self.terminated else None

    def terminate(self) -> None:
        """Mark the fake process terminated."""

        self.terminated = True

    def wait(self, timeout=None):
        """Return an exit status."""

        return 1


if __name__ == "__main__":
    unittest.main()
