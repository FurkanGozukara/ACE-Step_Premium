"""Unit tests for isolated training worker task helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.training.dataset_builder import AudioSample, DatasetBuilder
from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.training.subprocess_worker_tasks import (
    run_auto_label_task,
    run_lora_training_task,
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
                return [], {"value": "labeled"}, builder

            payload = {
                "dataset_path": str(dataset_path),
                "result_dataset_path": str(result_path),
                "dit_init_params": {"config_path": "model-a"},
                "llm_init_params": {"lm_model_path": "lm-a"},
                "settings": {"dataset_name": "worker-test"},
            }
            with patch(
                "acestep.ui.gradio.events.training.subprocess_worker_tasks.auto_label_all",
                side_effect=fake_auto_label,
            ):
                result = run_auto_label_task(payload, events.append)

        self.assertTrue(result["success"])
        self.assertEqual("labeled", result["status"])
        self.assertTrue(events)

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


def _save_dataset(path: Path) -> None:
    """Write a one-sample dataset for worker tests."""

    builder = DatasetBuilder()
    builder.samples = [
        AudioSample(
            audio_path=str(path.with_suffix(".wav")),
            filename="sample.wav",
            caption="caption",
            labeled=True,
        )
    ]
    status = builder.save_dataset(str(path), "worker-test")
    if not status.startswith("\u2705"):
        raise RuntimeError(status)


if __name__ == "__main__":
    unittest.main()
