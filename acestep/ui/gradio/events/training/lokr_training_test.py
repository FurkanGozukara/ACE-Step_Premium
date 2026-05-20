"""Tests for LoKr training handler setup."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.training.lokr_training import start_lokr_training


class LoKrTrainingHandlerTests(unittest.TestCase):
    """Verify LoKr training uses submitted Gradio values."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_start_training_uses_submitted_timestep_steps(self) -> None:
        """LoKr training configs should use the current Gradio schedule controls."""

        captured = {}

        class FakeTrainer:
            """Capture trainer config without running a real training loop."""

            def __init__(self, dit_handler, lokr_config, training_config) -> None:
                self.dit_handler = dit_handler
                captured["lokr_config"] = lokr_config
                captured["training_config"] = training_config

            def train_from_preprocessed(self, tensor_dir, training_state):
                """Return no training events after configuration is built."""

                return iter([])

        lightning_module = ModuleType("lightning")
        fabric_module = ModuleType("lightning.fabric")
        fabric_module.Fabric = object
        trainer_module = ModuleType("acestep.training.trainer")
        trainer_module.LoKRTrainer = FakeTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            output_dir = os.path.join(tmpdir, "lokr")
            handler = SimpleNamespace(model=object(), quantization=None, device="cpu")
            with patch.dict(
                sys.modules,
                {
                    "lightning": lightning_module,
                    "lightning.fabric": fabric_module,
                    "acestep.training.trainer": trainer_module,
                },
            ), patch(
                "acestep.ui.gradio.events.training.lokr_training._training_loss_figure",
                return_value=None,
            ):
                outputs = list(
                    start_lokr_training(
                        tmpdir,
                        handler,
                        32,
                        32,
                        8,
                        False,
                        False,
                        True,
                        False,
                        0.0002,
                        10,
                        1,
                        1,
                        5,
                        3.0,
                        42,
                        output_dir,
                        {},
                        training_num_inference_steps=51,
                    )
                )

        training_config = captured["training_config"]
        self.assertIn("LoKr training completed", outputs[-1][0])
        self.assertEqual(51, training_config.num_inference_steps)


if __name__ == "__main__":
    unittest.main()
