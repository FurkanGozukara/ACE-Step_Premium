"""Tests for SAM-Audio cancellation state."""

import unittest
from unittest.mock import MagicMock

from acestep.core.generation.cancellation import GenerationCancelled
from acestep.sam_audio_segment.cancel import (
    check_sam_audio_cancelled,
    register_sam_audio_subprocess,
    request_sam_audio_cancel,
    sam_audio_cancel_scope,
    unregister_sam_audio_subprocess,
)


class TestSamAudioCancel(unittest.TestCase):
    """Verify SAM-Audio scoped cancellation and subprocess termination."""

    def test_scope_request_raises_cancelled(self):
        """A cancellation request inside an active scope should be observable."""

        with sam_audio_cancel_scope():
            self.assertTrue(request_sam_audio_cancel())
            with self.assertRaises(GenerationCancelled):
                check_sam_audio_cancelled()

    def test_registered_subprocess_is_terminated(self):
        """Registered subprocesses should be terminated on cancel."""

        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = None

        register_sam_audio_subprocess(process)
        try:
            self.assertTrue(request_sam_audio_cancel())
            process.terminate.assert_called_once()
        finally:
            unregister_sam_audio_subprocess(process)


if __name__ == "__main__":
    unittest.main()
