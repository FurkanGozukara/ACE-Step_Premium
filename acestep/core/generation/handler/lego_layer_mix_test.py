"""Tests for Lego source-plus-layer waveform mixing."""

import unittest

import torch

from acestep.core.generation.handler.lego_layer_mix import apply_lego_layer_mix


class LegoLayerMixTests(unittest.TestCase):
    """Verify generated Lego layers are mixed over source audio."""

    def test_full_range_mixes_layer_over_source(self):
        """A full-song Lego range should keep source and add the generated layer."""

        source = torch.ones(1, 2, 4800) * 0.25
        layer = torch.ones(1, 2, 4800) * 0.5
        result = apply_lego_layer_mix(
            pred_wavs=layer,
            src_wavs=source,
            repainting_starts=[0.0],
            repainting_ends=[0.1],
            sample_rate=48000,
            crossfade_duration=0.0,
        )
        torch.testing.assert_close(result, torch.ones(1, 2, 4800) * 0.75)

    def test_negative_end_mixes_until_end(self):
        """A negative Lego range end should mean the layer continues to the end."""

        source = torch.ones(1, 2, 4800) * 0.25
        layer = torch.ones(1, 2, 4800) * 0.5
        result = apply_lego_layer_mix(
            pred_wavs=layer,
            src_wavs=source,
            repainting_starts=[0.0],
            repainting_ends=[-1.0],
            sample_rate=48000,
            crossfade_duration=0.0,
        )
        torch.testing.assert_close(result, torch.ones(1, 2, 4800) * 0.75)

    def test_bounded_range_preserves_source_outside_range(self):
        """Only the selected range should receive the generated layer."""

        source = torch.ones(1, 2, 9600) * 0.25
        layer = torch.ones(1, 2, 9600) * 0.5
        result = apply_lego_layer_mix(
            pred_wavs=layer,
            src_wavs=source,
            repainting_starts=[0.05],
            repainting_ends=[0.15],
            sample_rate=48000,
            crossfade_duration=0.0,
        )
        start = int(0.05 * 48000)
        end = int(0.15 * 48000)
        torch.testing.assert_close(result[0, :, :start], source[0, :, :start])
        torch.testing.assert_close(result[0, :, start:end], torch.ones(2, end - start) * 0.75)
        torch.testing.assert_close(result[0, :, end:], source[0, :, end:])

    def test_unbatched_source_audio_is_accepted(self):
        """The helper should accept a source tensor without a batch dimension."""

        source = torch.ones(2, 4800) * 0.25
        layer = torch.ones(1, 2, 4800) * 0.5
        result = apply_lego_layer_mix(
            pred_wavs=layer,
            src_wavs=source,
            repainting_starts=[0.0],
            repainting_ends=[0.1],
            sample_rate=48000,
            crossfade_duration=0.0,
        )
        torch.testing.assert_close(result, torch.ones(1, 2, 4800) * 0.75)


if __name__ == "__main__":
    unittest.main()
