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
    run_preprocess_subprocess,
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

    def test_auto_label_subprocess_sends_safe_roots_and_uses_one_worker(self) -> None:
        """A full auto-label batch should use one worker with source-audio roots."""

        with (
            tempfile.TemporaryDirectory() as project_dir,
            tempfile.TemporaryDirectory() as audio_dir,
        ):
            set_safe_roots([project_dir, audio_dir])
            label_dir = Path(project_dir) / "processed_labels"
            label_dir.mkdir()
            audio_paths = [Path(audio_dir) / "one.wav", Path(audio_dir) / "two.wav"]
            for path in audio_paths:
                path.write_bytes(b"audio")
            job = TrainingSubprocessJob(
                work_dir=Path(project_dir) / "job",
                request_path=Path(project_dir) / "job" / "request.json",
                result_path=Path(project_dir) / "job" / "result.json",
            )
            job.work_dir.mkdir(parents=True)
            captured_payloads: list[dict] = []

            def fake_stream(payload, _job):
                captured_payloads.append(payload)
                _builder_for_paths("new caption", audio_paths).save_dataset(
                    payload["result_dataset_path"],
                    "worker-test",
                )
                yield {
                    "kind": "result",
                    "result": {
                        "success": True,
                        "status": "done",
                        "dataset_path": payload["result_dataset_path"],
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
            ) as stream_worker:
                _table_update, status_update, returned = run_auto_label_subprocess(
                    builder_state=_builder_for_paths("old caption", audio_paths),
                    settings={
                        "dataset_name": "worker-test",
                        "label_output_dir": str(label_dir),
                    },
                    dit_init_params={"project_root": project_dir},
                    llm_init_params={},
                )

        self.assertEqual("done", status_update["value"])
        self.assertEqual(2, len(returned.samples))
        self.assertEqual(1, stream_worker.call_count)
        self.assertEqual(1, len(captured_payloads))
        safe_roots = {
            str(Path(root).resolve()) for root in captured_payloads[0]["safe_roots"]
        }
        self.assertIn(str(Path(audio_dir).resolve()), safe_roots)
        self.assertIn(str(label_dir.resolve()), safe_roots)
        self.assertEqual(
            str(Path(audio_dir).resolve()),
            str(Path(captured_payloads[0]["settings"]["label_source_root"]).resolve()),
        )

    def test_preprocess_subprocess_sends_safe_roots_and_uses_one_worker(self) -> None:
        """A full preprocess batch should use one worker with source-audio roots."""

        with (
            tempfile.TemporaryDirectory() as project_dir,
            tempfile.TemporaryDirectory() as audio_dir,
        ):
            set_safe_roots([project_dir, audio_dir])
            audio_paths = [Path(audio_dir) / "one.wav", Path(audio_dir) / "two.wav"]
            for path in audio_paths:
                path.write_bytes(b"audio")
            job = TrainingSubprocessJob(
                work_dir=Path(project_dir) / "job",
                request_path=Path(project_dir) / "job" / "request.json",
                result_path=Path(project_dir) / "job" / "result.json",
            )
            job.work_dir.mkdir(parents=True)
            captured_payloads: list[dict] = []

            def fake_stream(payload, _job):
                captured_payloads.append(payload)
                yield {
                    "kind": "result",
                    "result": {"success": True, "status": "preprocessed"},
                }

            with patch(
                "acestep.ui.gradio.events.training.subprocess_dataset."
                "create_training_subprocess_job",
                return_value=job,
            ), patch(
                "acestep.ui.gradio.events.training.subprocess_dataset."
                "stream_training_subprocess_job",
                side_effect=fake_stream,
            ) as stream_worker:
                status = run_preprocess_subprocess(
                    output_dir=str(Path(project_dir) / "tensors"),
                    preprocess_mode="lora",
                    builder_state=_builder_for_paths("caption", audio_paths),
                    model_config="model-a",
                    dit_init_params={"project_root": project_dir},
                )

        self.assertEqual("preprocessed", status)
        self.assertEqual(1, stream_worker.call_count)
        self.assertEqual(1, len(captured_payloads))
        safe_roots = {
            str(Path(root).resolve()) for root in captured_payloads[0]["safe_roots"]
        }
        self.assertIn(str(Path(audio_dir).resolve()), safe_roots)


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


def _builder_for_paths(caption: str, audio_paths: list[Path]) -> DatasetBuilder:
    """Build a labeled dataset builder for the given audio paths."""

    builder = DatasetBuilder()
    if audio_paths:
        builder._current_dir = str(audio_paths[0].parent)
    builder.samples = [
        AudioSample(
            audio_path=str(audio_path),
            filename=audio_path.name,
            caption=caption,
            labeled=True,
        )
        for audio_path in audio_paths
    ]
    return builder


if __name__ == "__main__":
    unittest.main()
