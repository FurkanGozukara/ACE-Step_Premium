"""Unit tests for subprocess generation streaming."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.core.generation.cancellation import (
    GenerationCancelled,
    generation_cancel_scope,
    request_generation_cancel,
)
from acestep.ui.gradio.events.results.subprocess_generation import (
    stream_subprocess_generation,
)


class _CancellingStdout:
    """Stdout stub that requests cancellation while the parent is reading."""

    def readline(self) -> str:
        """Cancel the registered subprocess and then signal EOF."""

        request_generation_cancel(subprocess_only=True)
        return ""

    def readlines(self) -> list[str]:
        """Return no remaining worker output."""

        return []


class _FakeProcess:
    """Minimal ``Popen`` stand-in for subprocess cancellation tests."""

    def __init__(self) -> None:
        """Initialize fake process state."""

        self.stdout = _CancellingStdout()
        self.terminated = False
        self.killed = False

    def poll(self):
        """Return process status."""

        return 1 if self.terminated or self.killed else None

    def terminate(self) -> None:
        """Record graceful termination."""

        self.terminated = True

    def wait(self, timeout=None):
        """Return the fake process return code."""

        _ = timeout
        return 1


class SubprocessGenerationTests(unittest.TestCase):
    """Verify parent streaming behavior around isolated workers."""

    def tearDown(self) -> None:
        """Clear cancellation state left by each test."""

        with generation_cancel_scope():
            pass

    def test_cancel_during_stdout_read_surfaces_as_generation_cancelled(self) -> None:
        """Cancellation during stdout read should not become a generic failure."""

        with tempfile.TemporaryDirectory() as tmp:
            payload = {"project_root": tmp, "service": {}, "generation": {}}
            process = _FakeProcess()

            with patch(
                "acestep.ui.gradio.events.results.subprocess_generation.subprocess.Popen",
                return_value=process,
            ) as popen:
                generator = stream_subprocess_generation(payload)
                self.assertEqual(next(generator)["kind"], "status")
                self.assertIsNone(popen.call_args.kwargs["stderr"])
                with self.assertRaises(GenerationCancelled):
                    next(generator)

            self.assertTrue(process.terminated)
            self.assertTrue((Path(tmp) / ".cache" / "acestep" / "subprocess_jobs").exists())


if __name__ == "__main__":
    unittest.main()
