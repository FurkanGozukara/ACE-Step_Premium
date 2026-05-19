"""Unit tests for isolated training worker task helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.training.dataset_builder import AudioSample, DatasetBuilder
from acestep.training.dataset_builder_modules.label_persistence import (
    save_sample_label_metadata,
)
from acestep.training.path_safety import get_safe_roots, safe_path, set_safe_roots
from acestep.ui.gradio.events.training.subprocess_worker_tasks import (
    run_auto_label_task,
    run_lora_training_task,
    run_preprocess_task,
)


class SubprocessWorkerTaskTests(unittest.TestCase):
    """Verify worker tasks seed handlers and return serializable results."""

    def setUp(self) -> None:
        """Preserve training safe roots."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore training safe roots."""

        set_safe_roots(self._safe_roots)

    def test_auto_label_task_seeds_parent_init_params(self) -> None:
        """Auto-label worker should pass serialized init params into fresh handlers."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            dataset_path = Path(tmpdir) / "dataset.json"
            result_path = Path(tmpdir) / "result.json"
            _save_dataset(dataset_path)
            events: list[dict] = []

            def fake_auto_label(dit, llm, builder, *args, **kwargs):
                self.assertEqual("model-a", dit.last_init_params["config_path"])
                self.assertEqual("lm-a", llm.last_init_params["lm_model_path"])
                self.assertEqual(str(Path(tmpdir) / "labels"), kwargs["label_output_dir"])
                return [], {"value": "labeled"}, builder

            payload = {
                "dataset_path": str(dataset_path),
                "result_dataset_path": str(result_path),
                "dit_init_params": {"config_path": "model-a"},
                "llm_init_params": {"lm_model_path": "lm-a"},
                "settings": {
                    "dataset_name": "worker-test",
                    "label_output_dir": str(Path(tmpdir) / "labels"),
                },
            }
            with patch(
                "acestep.ui.gradio.events.training.subprocess_worker_tasks.auto_label_all",
                side_effect=fake_auto_label,
            ):
                result = run_auto_label_task(payload, events.append)

        self.assertTrue(result["success"])
        self.assertEqual("labeled", result["status"])
        self.assertTrue(events)

    def test_auto_label_task_applies_payload_safe_roots_for_processed_labels(self) -> None:
        """Auto-label worker should write labels outside the source audio folder."""

        with (
            tempfile.TemporaryDirectory() as project_dir,
            tempfile.TemporaryDirectory() as audio_dir,
        ):
            dataset_path = Path(project_dir) / "dataset.json"
            result_path = Path(project_dir) / "result.json"
            label_dir = Path(project_dir) / "processed_labels"
            audio_path = Path(audio_dir) / "sample.wav"
            audio_path.write_bytes(b"audio")
            set_safe_roots([project_dir])
            _save_dataset(dataset_path, audio_path=audio_path)
            events: list[dict] = []

            def fake_auto_label(_dit, _llm, builder, *args, **kwargs):
                sample = builder.samples[0]
                sample.caption = "caption"
                sample.labeled = True
                save_sample_label_metadata(
                    sample,
                    output_dir=kwargs["label_output_dir"],
                    source_root=kwargs["label_source_root"],
                )
                return [], {"value": "labeled"}, builder

            payload = {
                "dataset_path": str(dataset_path),
                "result_dataset_path": str(result_path),
                "dit_init_params": {},
                "llm_init_params": {},
                "safe_roots": [project_dir, audio_dir, str(label_dir)],
                "settings": {
                    "dataset_name": "worker-test",
                    "label_output_dir": str(label_dir),
                    "label_source_root": audio_dir,
                },
            }
            with patch(
                "acestep.ui.gradio.events.training.subprocess_worker_tasks.auto_label_all",
                side_effect=fake_auto_label,
            ):
                result = run_auto_label_task(payload, events.append)

            label_exists = (label_dir / "sample.json").exists()
            source_sidecar_exists = audio_path.with_suffix(".json").exists()

        self.assertTrue(result["success"])
        self.assertTrue(label_exists)
        self.assertFalse(source_sidecar_exists)

    def test_preprocess_task_applies_payload_safe_roots(self) -> None:
        """Preprocess worker should validate output and source-audio roots."""

        with (
            tempfile.TemporaryDirectory() as project_dir,
            tempfile.TemporaryDirectory() as audio_dir,
        ):
            dataset_path = Path(project_dir) / "dataset.json"
            output_dir = Path(audio_dir) / "tensors"
            audio_path = Path(audio_dir) / "sample.wav"
            audio_path.write_bytes(b"audio")
            set_safe_roots([project_dir])
            _save_dataset(dataset_path, audio_path=audio_path)
            events: list[dict] = []

            def fake_preprocess(output_dir_arg, _mode, _dit, builder, **_kwargs):
                self.assertEqual(str(output_dir.resolve()), safe_path(output_dir_arg))
                self.assertEqual(
                    str(audio_path.resolve()),
                    safe_path(builder.samples[0].audio_path),
                )
                return "preprocessed"

            payload = {
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "dit_init_params": {},
                "safe_roots": [project_dir, audio_dir],
            }
            with patch(
                "acestep.ui.gradio.events.training.subprocess_worker_tasks.preprocess_dataset",
                side_effect=fake_preprocess,
            ):
                result = run_preprocess_task(payload, events.append)

        self.assertTrue(result["success"])
        self.assertEqual("preprocessed", result["status"])

    def test_lora_training_task_emits_training_events(self) -> None:
        """LoRA worker should stream each training handler yield."""

        events: list[dict] = []

        def fake_start_training(**kwargs):
            self.assertEqual("model-b", kwargs["dit_handler"].last_init_params["config_path"])
            yield "Epoch 1/2, Step 1, Loss: 0.5", "log", None, {"is_training": True}

        payload = {
            "dit_init_params": {"config_path": "model-b"},
            "training_args": {"tensor_dir": "tensors"},
            "training_state": {},
        }
        with patch(
            "acestep.ui.gradio.events.training.subprocess_worker_tasks.start_training",
            side_effect=fake_start_training,
        ):
            result = run_lora_training_task(payload, events.append)

        self.assertTrue(result["success"])
        self.assertEqual("training", events[0]["kind"])
        self.assertIn("Loss", events[0]["status"])


def _save_dataset(path: Path, audio_path: Path | None = None) -> None:
    """Write a one-sample dataset for worker tests."""

    builder = DatasetBuilder()
    builder.samples = [
        AudioSample(
            audio_path=str(audio_path or path.with_suffix(".wav")),
            filename=(audio_path.name if audio_path else "sample.wav"),
            caption="caption",
            labeled=True,
        )
    ]
    status = builder.save_dataset(str(path), "worker-test")
    if not status.startswith("\u2705"):
        raise RuntimeError(status)


if __name__ == "__main__":
    unittest.main()
