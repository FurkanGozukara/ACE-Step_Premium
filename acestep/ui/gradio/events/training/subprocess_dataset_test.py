"""Unit tests for parent-side dataset subprocess helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.training.dataset_builder import AudioSample, DatasetBuilder
from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.training.subprocess_dataset import (
    run_auto_label_subprocess,
)
from acestep.ui.gradio.events.training.subprocess_runner import TrainingSubprocessJob


class SubprocessDatasetTests(unittest.TestCase):
    """Verify parent helpers serialize and reload dataset state."""

    def setUp(self) -> None:
        """Preserve training safe roots."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore training safe roots."""

        set_safe_roots(self._safe_roots)

    def test_auto_label_subprocess_loads_worker_result_builder(self) -> None:
        """Parent auto-label helper should return the worker-updated builder."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            job = TrainingSubprocessJob(
                work_dir=Path(tmpdir),
                request_path=Path(tmpdir) / "request.json",
                result_path=Path(tmpdir) / "result.json",
            )
            result_dataset = Path(tmpdir) / "auto_label_result.json"
            _builder("new caption").save_dataset(str(result_dataset), "worker-test")

            def fake_stream(payload, _job):
                self.assertEqual(str(result_dataset), payload["result_dataset_path"])
                yield {"kind": "status", "message": "half done"}
                yield {
                    "kind": "result",
                    "result": {
                        "success": True,
                        "status": "done",
                        "dataset_path": str(result_dataset),
                    },
                }

            with patch(
                "acestep.ui.gradio.events.training.subprocess_dataset."
                "create_training_subprocess_job",
                return_value=job,
            ), patch(
                "acestep.ui.gradio.events.training.subprocess_dataset."
                "stream_training_subprocess_job",
                side_effect=fake_stream,
            ):
                table_update, status_update, returned = run_auto_label_subprocess(
                    builder_state=_builder("old caption"),
                    settings={"dataset_name": "worker-test"},
                    dit_init_params={"project_root": tmpdir},
                    llm_init_params={},
                )

        self.assertEqual("done", status_update["value"])
        self.assertEqual("new caption", returned.samples[0].caption)
        self.assertIn("sample.wav", table_update["value"][0])


def _builder(caption: str) -> DatasetBuilder:
    """Build a one-sample dataset builder."""

    builder = DatasetBuilder()
    builder.samples = [
        AudioSample(
            audio_path="sample.wav",
            filename="sample.wav",
            caption=caption,
            labeled=True,
        )
    ]
    return builder


if __name__ == "__main__":
    unittest.main()
