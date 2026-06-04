"""Tests for SAM-Audio fast initialization helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from acestep.sam_audio_segment.initialization import (
    checkpoint_skip_prefixes_for_settings,
    fast_checkpoint_model_initialization,
    should_skip_visual_encoder,
    skip_default_parameter_initialization,
)
from acestep.sam_audio_segment.settings import SamAudioSettings


class InitializationTests(unittest.TestCase):
    """Verify temporary parameter-init suppression is scoped."""

    def test_skip_default_parameter_initialization_restores_reset(self) -> None:
        """Layer reset hooks should be skipped only inside the context."""

        calls = []

        def _record_reset(_module: torch.nn.Module) -> None:
            calls.append("reset")

        with patch.object(torch.nn.Linear, "reset_parameters", _record_reset):
            with skip_default_parameter_initialization():
                torch.nn.Linear(2, 2)
            torch.nn.Linear(2, 2)

        self.assertEqual(["reset"], calls)

    def test_text_modes_skip_visual_checkpoint_prefixes(self) -> None:
        """Only visual prompting should require visual encoder weights."""

        text_settings = SamAudioSettings(prompt_mode="text")
        span_settings = SamAudioSettings(prompt_mode="span")
        visual_settings = SamAudioSettings(prompt_mode="visual")

        self.assertTrue(should_skip_visual_encoder(text_settings))
        self.assertTrue(should_skip_visual_encoder(span_settings))
        self.assertFalse(should_skip_visual_encoder(visual_settings))
        self.assertEqual(
            ("vision_encoder.",),
            checkpoint_skip_prefixes_for_settings(text_settings),
        )
        self.assertEqual((), checkpoint_skip_prefixes_for_settings(visual_settings))

    def test_fast_initialization_patches_and_restores_visual_encoder(self) -> None:
        """Visual encoder replacement should be scoped to the fast-load context."""

        original_encoder = object()
        fake_module = SimpleNamespace(PerceptionEncoder=original_encoder)

        with patch(
            "acestep.sam_audio_segment.initialization.import_module",
            return_value=fake_module,
        ):
            with fast_checkpoint_model_initialization(skip_visual_encoder=True):
                self.assertIsNot(original_encoder, fake_module.PerceptionEncoder)
            self.assertIs(original_encoder, fake_module.PerceptionEncoder)


if __name__ == "__main__":
    unittest.main()
