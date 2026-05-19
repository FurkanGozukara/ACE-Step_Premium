"""Tests for LoRA training-step estimate helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.training.step_estimate import (
    format_lora_step_estimate,
)


class LoraStepEstimateTests(unittest.TestCase):
    """Verify step estimates from tensor dataset size and training controls."""

    def setUp(self) -> None:
        """Preserve global safe-root state."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore global safe-root state."""

        set_safe_roots(self._safe_roots)

    def test_estimate_uses_manifest_sample_count(self) -> None:
        """Manifest sample count should drive total optimizer updates."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            dataset_dir = Path(tmpdir) / "tensors"
            dataset_dir.mkdir()
            (dataset_dir / "manifest.json").write_text(
                json.dumps({"num_samples": 50}),
                encoding="utf-8",
            )

            estimate = format_lora_step_estimate(dataset_dir, 2, 4, 100)

        self.assertIn("700 optimizer updates", estimate)
        self.assertIn("ceil(ceil(50 / 2) / 4) * 100", estimate)
        self.assertIn("2 * 4 = 8", estimate)

    def test_estimate_counts_pt_files_without_manifest(self) -> None:
        """Datasets without a manifest should still estimate from .pt files."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            dataset_dir = Path(tmpdir) / "tensors"
            dataset_dir.mkdir()
            for index in range(5):
                (dataset_dir / f"{index}.pt").write_bytes(b"")

            estimate = format_lora_step_estimate(dataset_dir, 2, 2, 3)

        self.assertIn("6 optimizer updates", estimate)

    def test_estimate_prompts_when_dataset_is_missing(self) -> None:
        """Missing or unsafe directories should not raise from the UI helper."""

        estimate = format_lora_step_estimate("", 1, 1, 100)

        self.assertEqual("Load a tensor dataset to calculate total training steps.", estimate)


if __name__ == "__main__":
    unittest.main()
