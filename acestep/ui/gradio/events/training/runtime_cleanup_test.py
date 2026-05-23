"""Tests for parent-process cleanup before LoRA training starts."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn

from acestep.ui.gradio.events.training.runtime_cleanup import (
    prepare_parent_runtime_for_training,
)


class _FakeLlmHandler:
    """Small initialized LM stand-in with an unload hook."""

    def __init__(self) -> None:
        self.llm_initialized = True
        self.unload = MagicMock(side_effect=self._mark_unloaded)

    def _mark_unloaded(self) -> None:
        """Mark the fake LM as unloaded."""

        self.llm_initialized = False


class _FakeDitHandler:
    """Small DiT stand-in that supports full runtime release."""

    def __init__(self) -> None:
        self.model = object()
        self.vae = object()
        self.text_encoder = object()
        self.release_calls = 0

    def _release_loaded_runtime_components(self) -> None:
        """Release fake model handles like the real generation handler."""

        self.release_calls += 1
        self.model = None
        self.vae = None
        self.text_encoder = None


class _InlineDitHandler:
    """Small handler with decoder and generation-only modules."""

    def __init__(self) -> None:
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.model = type(
            "FakeModel",
            (),
            {
                "decoder": nn.Linear(2, 2),
                "music_encoder": nn.Linear(2, 2),
                "vae": nn.Linear(2, 2),
            },
        )()
        self.release_calls = 0

    def _release_loaded_runtime_components(self) -> None:
        """Track unexpected full releases during inline training cleanup."""

        self.release_calls += 1


class RuntimeCleanupTests(unittest.TestCase):
    """Verify cleanup behavior for subprocess and inline training starts."""

    def test_subprocess_cleanup_unloads_lm_and_releases_parent_dit(self) -> None:
        """Subprocess training should free parent generation models first."""

        dit_handler = _FakeDitHandler()
        llm_handler = _FakeLlmHandler()

        with patch(
            "acestep.ui.gradio.events.training.runtime_cleanup.cleanup_runtime_memory"
        ) as cleanup:
            status = prepare_parent_runtime_for_training(
                dit_handler,
                llm_handler,
                release_dit=True,
            )

        self.assertIn("5Hz LM", status)
        self.assertIn("DiT generation runtime", status)
        llm_handler.unload.assert_called_once_with()
        self.assertFalse(llm_handler.llm_initialized)
        self.assertEqual(1, dit_handler.release_calls)
        self.assertIsNone(dit_handler.model)
        cleanup.assert_called_once_with()

    def test_inline_cleanup_keeps_dit_runtime_and_offloads_unused_modules(self) -> None:
        """Inline training should keep the decoder resident and only offload helpers."""

        dit_handler = _InlineDitHandler()

        with patch(
            "acestep.ui.gradio.events.training.runtime_cleanup.cleanup_runtime_memory"
        ) as cleanup:
            status = prepare_parent_runtime_for_training(
                dit_handler,
                None,
                release_dit=False,
            )

        self.assertIn("vae to CPU", status)
        self.assertIn("text_encoder to CPU", status)
        self.assertEqual(0, dit_handler.release_calls)
        self.assertIsNotNone(dit_handler.model.decoder)
        cleanup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
