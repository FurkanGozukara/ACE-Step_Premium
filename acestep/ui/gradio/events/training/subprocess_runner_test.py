"""Integration tests for isolated training subprocess launching."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.training.subprocess_runner import (
    TrainingSubprocessJob,
    stream_training_subprocess_job,
)


class SubprocessRunnerTests(unittest.TestCase):
    """Verify the parent runner executes the worker as a real subprocess."""

    def test_worker_failure_result_is_reported(self) -> None:
        """An unknown worker operation should round-trip as a RuntimeError."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path.cwd()
            job = TrainingSubprocessJob(
                work_dir=Path(tmpdir),
                request_path=Path(tmpdir) / "request.json",
                result_path=Path(tmpdir) / "result.json",
            )
            payload = {"operation": "unknown", "project_root": str(root)}

            with self.assertRaisesRegex(RuntimeError, "Unknown worker operation"):
                list(stream_training_subprocess_job(payload, job))


if __name__ == "__main__":
    unittest.main()
