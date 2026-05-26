"""Tests for LoRA/DoRA best-checkpoint saving."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from acestep.training.configs import TrainingConfig
from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.training.trainer import _save_best_lora_checkpoint


class SaveBestCheckpointTests(unittest.TestCase):
    """Verify best checkpoint replacement behavior."""

    def setUp(self) -> None:
        """Preserve safe roots."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe roots."""

        set_safe_roots(self._safe_roots)

    def test_save_best_replaces_existing_best_directory(self) -> None:
        """Saving a new best should clear the previous best directory first."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            best_dir = os.path.join(tmpdir, "best")
            os.makedirs(best_dir)
            stale_file = os.path.join(best_dir, "old.safetensors")
            with open(stale_file, "w", encoding="utf-8") as handle:
                handle.write("stale")
            config = TrainingConfig(output_dir=tmpdir, lora_name="song")

            with patch(
                "acestep.training.trainer.save_training_checkpoint",
                return_value=os.path.join(best_dir, "song-best-training_resume_state.pt"),
            ) as save_checkpoint:
                state_path = _save_best_lora_checkpoint(
                    Mock(),
                    Mock(),
                    Mock(),
                    config,
                    epoch=3,
                    global_step=9,
                )

            self.assertFalse(os.path.exists(stale_file))
            self.assertTrue(state_path.endswith("song-best-training_resume_state.pt"))
            save_checkpoint.assert_called_once()
            self.assertEqual(best_dir, save_checkpoint.call_args.args[5])
            self.assertEqual("song-best", save_checkpoint.call_args.kwargs["artifact_name"])


if __name__ == "__main__":
    unittest.main()
