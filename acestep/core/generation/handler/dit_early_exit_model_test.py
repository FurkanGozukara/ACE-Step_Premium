"""Tests for DiT alignment early-exit behavior."""

from __future__ import annotations

import sys
import unittest

import torch

# Transformers auto_docstring may print Unicode during model import on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from acestep.models.turbo.configuration_acestep_v15 import AceStepConfig
from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepDiTModel


def _tiny_config() -> AceStepConfig:
    """Return a minimal DiT config that can run in CPU unit tests."""

    return AceStepConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        audio_acoustic_hidden_dim=4,
        in_channels=12,
        patch_size=2,
        use_sliding_window=False,
        layer_types=["full_attention"] * 4,
        text_hidden_dim=8,
        timbre_hidden_dim=4,
        _attn_implementation="eager",
    )


def _decoder_inputs() -> dict[str, torch.Tensor]:
    """Return shape-compatible inputs for the tiny DiT decoder."""

    return {
        "hidden_states": torch.randn(1, 8, 4),
        "timestep": torch.ones(1),
        "timestep_r": torch.ones(1),
        "attention_mask": torch.ones(1, 8),
        "encoder_hidden_states": torch.randn(1, 5, 16),
        "encoder_attention_mask": torch.ones(1, 5),
        "context_latents": torch.randn(1, 8, 8),
    }


def _record_layer_calls(model: AceStepDiTModel) -> list[int]:
    """Wrap layer forwards and return the call-order sink."""

    called: list[int] = []
    for index, layer in enumerate(model.layers):
        original_forward = layer.forward

        def _wrapped(*args, _index=index, _forward=original_forward, **kwargs):
            called.append(_index)
            return _forward(*args, **kwargs)

        layer.forward = _wrapped
    return called


class DiTEarlyExitTests(unittest.TestCase):
    """Verify alignment-only decoding avoids full DiT execution."""

    def test_alignment_early_exit_stops_after_highest_configured_layer(self) -> None:
        """Auto-LRC attention extraction should stop after the requested layer."""

        model = AceStepDiTModel(_tiny_config()).eval()
        called = _record_layer_calls(model)

        with torch.inference_mode():
            output = model(
                **_decoder_inputs(),
                output_attentions=True,
                custom_layers_config={1: [0]},
                enable_early_exit=True,
            )

        self.assertEqual([0, 1], called)
        self.assertEqual(3, len(output))
        self.assertEqual(2, len(output[2]))

    def test_normal_decode_still_runs_all_layers(self) -> None:
        """Regular generation must keep the full decoder path unchanged."""

        model = AceStepDiTModel(_tiny_config()).eval()
        called = _record_layer_calls(model)

        with torch.inference_mode():
            output = model(**_decoder_inputs())

        self.assertEqual([0, 1, 2, 3], called)
        self.assertEqual((1, 8, 4), tuple(output[0].shape))


if __name__ == "__main__":
    unittest.main()
