"""Tests for LoRA training run wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.training_run_wiring import _build_training_wrapper


class TrainingRunWiringTests(unittest.TestCase):
    """Verify LoRA training wrapper argument forwarding."""

    def test_training_wrapper_passes_selected_model_config(self) -> None:
        """The LoRA base-model dropdown value should reach the training handler."""

        def fake_start_training(*_args, **kwargs):
            """Yield one result while exposing call kwargs through the patch mock."""

            self.assertEqual("model-b", kwargs["model_config"])
            yield "started", "", None, {"is_training": True}

        wrapper = _build_training_wrapper(dit_handler=object())

        with patch(
            "acestep.ui.gradio.events.wiring.training_run_wiring.train_h.start_training",
            side_effect=fake_start_training,
        ):
            outputs = list(
                wrapper(
                    "tensors",
                    64,
                    128,
                    0.1,
                    0.0003,
                    10,
                    1,
                    1,
                    10,
                    3.0,
                    42,
                    "out",
                    "",
                    "model-b",
                    {},
                )
            )

        self.assertEqual("started", outputs[0][0])


if __name__ == "__main__":
    unittest.main()
