"""Tests for LoRA training wrapper parent-runtime cleanup."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from acestep.ui.gradio.events.wiring.training_lora_run_wrapper import (
    build_lora_training_wrapper,
)


_WRAPPER_MODULE = "acestep.ui.gradio.events.wiring.training_lora_run_wrapper"


class TrainingLoraRunCleanupTests(unittest.TestCase):
    """Verify cleanup runs before LoRA training starts."""

    def test_subprocess_training_snapshots_init_before_parent_cleanup(self) -> None:
        """The worker payload should be captured before releasing parent DiT state."""

        call_order: list[str] = []
        dit_handler = object()
        llm_handler = object()
        wrapper = build_lora_training_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            normalize_training_state=_normalize_training_state,
        )

        def build_payload(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            call_order.append("build")
            return {"project_root": "."}

        def cleanup(*_args: Any, **_kwargs: Any) -> str:
            call_order.append("cleanup")
            return "cleanup complete"

        with patch(
            f"{_WRAPPER_MODULE}.build_dit_init_payload",
            side_effect=build_payload,
        ), patch(
            f"{_WRAPPER_MODULE}.prepare_parent_runtime_for_training",
            side_effect=cleanup,
        ) as cleanup_mock, patch(
            f"{_WRAPPER_MODULE}.stream_lora_training_subprocess",
            return_value=iter([("trained", "log", None, {"is_training": False})]),
        ):
            outputs = list(_call_wrapper(wrapper, training_subprocess=True))

        self.assertEqual(["build", "cleanup"], call_order)
        cleanup_mock.assert_called_once_with(
            dit_handler,
            llm_handler,
            release_dit=True,
        )
        self.assertEqual("cleanup complete", outputs[0][0])
        self.assertEqual("trained", outputs[1][0])

    def test_inline_training_keeps_parent_dit_for_training(self) -> None:
        """Inline training should clean up without fully releasing parent DiT."""

        dit_handler = object()
        wrapper = build_lora_training_wrapper(
            dit_handler=dit_handler,
            normalize_training_state=_normalize_training_state,
        )

        with patch(
            f"{_WRAPPER_MODULE}.prepare_parent_runtime_for_training",
            return_value="inline cleanup",
        ) as cleanup_mock, patch(
            f"{_WRAPPER_MODULE}.train_h.start_training",
            return_value=iter([("started", "", None, {"is_training": True})]),
        ):
            outputs = list(_call_wrapper(wrapper, training_subprocess=False))

        cleanup_mock.assert_called_once_with(
            dit_handler,
            None,
            release_dit=False,
        )
        self.assertEqual("inline cleanup", outputs[0][0])
        self.assertEqual("started", outputs[1][0])


def _call_wrapper(wrapper: Any, *, training_subprocess: bool) -> Any:
    """Call a LoRA wrapper with the standard positional argument contract."""

    return wrapper(
        "tensors",
        "test-lora",
        16,
        32,
        0.0,
        0.0001,
        1,
        1,
        1,
        0,
        3.0,
        8,
        1234,
        "out",
        "",
        True,
        False,
        True,
        True,
        False,
        "Disabled",
        1,
        False,
        10,
        "prompt",
        "lyrics",
        1234,
        False,
        True,
        training_subprocess,
        "model-b",
        "Manual",
        {"is_training": False, "should_stop": False},
    )


def _normalize_training_state(training_state: Any) -> dict[str, bool]:
    """Return a valid mutable training state."""

    if isinstance(training_state, dict):
        return training_state
    return {"is_training": False, "should_stop": False}


if __name__ == "__main__":
    unittest.main()
