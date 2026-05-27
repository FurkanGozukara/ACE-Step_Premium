"""Tests for LoRA step-estimate wiring helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.wiring.training_step_estimate_wiring import (
    format_lora_training_dataset_info,
    reset_lora_loaded_dataset,
)


class TrainingStepEstimateWiringTests(unittest.TestCase):
    """Verify loaded-dataset gated estimate and validation preview helpers."""

    def setUp(self) -> None:
        """Preserve global safe-root state."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore global safe-root state."""

        set_safe_roots(self._safe_roots)

    def test_validation_slider_without_loaded_dataset_returns_prompts(self) -> None:
        """Changing validation split before loading should not scan a path."""

        estimate, validation = format_lora_training_dataset_info("", 1, 1, 10, 50)

        self.assertEqual(
            "Load a tensor dataset to calculate total training steps.",
            estimate,
        )
        self.assertEqual(
            "Load a tensor dataset to preview the validation split.",
            validation,
        )

    def test_loaded_dataset_updates_estimate_and_validation_preview(self) -> None:
        """Loaded tensor state should drive both dataset-dependent messages."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            dataset_dir = Path(tmpdir) / "tensors"
            dataset_dir.mkdir()
            (dataset_dir / "manifest.json").write_text(
                json.dumps({"num_samples": 20}),
                encoding="utf-8",
            )

            estimate, validation = format_lora_training_dataset_info(
                dataset_dir,
                2,
                2,
                10,
                10,
            )

        self.assertIn("50 optimizer updates", estimate)
        self.assertIn("Training samples: `18`", validation)
        self.assertIn("Validation samples: `2`", validation)

    def test_tensor_path_edit_resets_loaded_dataset_state(self) -> None:
        """Editing the tensor textbox should clear active loaded-dataset state."""

        loaded_dir, estimate, validation = reset_lora_loaded_dataset()

        self.assertEqual("", loaded_dir)
        self.assertEqual(
            "Load a tensor dataset to calculate total training steps.",
            estimate,
        )
        self.assertEqual(
            "Load a tensor dataset to preview the validation split.",
            validation,
        )


if __name__ == "__main__":
    unittest.main()
