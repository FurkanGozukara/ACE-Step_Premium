"""Tests for LoRA training run wiring."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.training_lora_run_wrapper import (
    build_lora_training_wrapper,
)


class TrainingRunWiringTests(unittest.TestCase):
    """Verify LoRA training wrapper argument forwarding."""

    def test_training_wrapper_passes_selected_model_config(self) -> None:
        """The LoRA base-model dropdown value should reach the training handler."""

        def fake_start_training(*_args, **kwargs):
            """Yield one result while exposing call kwargs through the patch mock."""

            self.assertEqual("my-awesome-song", kwargs["lora_name"])
            self.assertEqual("model-b", kwargs["model_config"])
            self.assertTrue(kwargs["gradient_checkpointing"])
            self.assertEqual("Disabled", kwargs["base_quantization"])
            self.assertFalse(kwargs["sample_generation_enabled"])
            self.assertEqual("prompt", kwargs["sample_prompt"])
            self.assertEqual("model-b", kwargs["sample_generation_model_config"])
            self.assertEqual(
                {"config_path": "model-b", "guidance_scale": 1.0},
                kwargs["sample_generation_settings"],
            )
            yield "started", "", None, {"is_training": True}

        wrapper = build_lora_training_wrapper(
            dit_handler=object(),
            normalize_training_state=_normalize_training_state,
            sample_setting_keys=("config_path", "guidance_scale"),
        )

        with patch(
            "acestep.ui.gradio.events.wiring.training_lora_run_wrapper.train_h.start_training",
            side_effect=fake_start_training,
        ):
            outputs = list(
                wrapper(
                    "tensors",
                    "my-awesome-song",
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
                    True,
                    False,
                    True,
                    True,
                    True,
                    "Disabled",
                    10,
                    False,
                    10,
                    "prompt",
                    "lyrics",
                    30,
                    8,
                    42,
                    "samples",
                    True,
                    True,
                    False,
                    "model-b",
                    "Manual",
                    {"is_training": False, "should_stop": False},
                    "model-b",
                    1.0,
                )
            )

        self.assertEqual("started", outputs[0][0])

    def test_training_wrapper_streams_subprocess_when_enabled(self) -> None:
        """The subprocess checkbox should route LoRA training through the worker stream."""

        wrapper = build_lora_training_wrapper(
            dit_handler=object(),
            normalize_training_state=_normalize_training_state,
        )

        with patch(
            "acestep.ui.gradio.events.wiring.training_lora_run_wrapper.build_dit_init_payload",
            return_value={"project_root": "."},
        ) as build_init, patch(
            "acestep.ui.gradio.events.wiring.training_lora_run_wrapper.stream_lora_training_subprocess",
            return_value=iter([("subprocess", "log", None, {"is_training": False})]),
        ) as stream:
            outputs = list(
                wrapper(
                    "tensors",
                    "my-awesome-song",
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
                    True,
                    False,
                    True,
                    True,
                    True,
                    "Disabled",
                    10,
                    False,
                    10,
                    "prompt",
                    "lyrics",
                    30,
                    8,
                    42,
                    "samples",
                    True,
                    True,
                    True,
                    "model-b",
                    "Manual",
                    {"is_training": False, "should_stop": False},
                )
            )

        self.assertEqual("subprocess", outputs[0][0])
        build_init.assert_called_once()
        self.assertTrue(stream.call_args.kwargs["training_args"]["gradient_checkpointing"])


def _normalize_training_state(training_state: Any) -> dict[str, bool]:
    """Return a valid training state for wrapper tests."""

    if isinstance(training_state, dict):
        return training_state
    return {"is_training": False, "should_stop": False}


if __name__ == "__main__":
    unittest.main()
