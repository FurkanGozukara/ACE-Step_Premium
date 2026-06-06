"""Tests for DiffPitcher diffusion loop progress reporting."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from acestep.audio_processing.diffpitcher_inference_loop import run_diffusion


class _FakeStepResult:
    """Minimal scheduler result carrying the updated sample tensor."""

    def __init__(self, prev_sample: torch.Tensor) -> None:
        self.prev_sample = prev_sample


class _FakeScheduler:
    """Small DDIMScheduler stand-in for deterministic progress tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.timesteps: list[torch.Tensor] = []

    def set_timesteps(self, steps: int, device: torch.device) -> None:
        """Create the requested number of fake scheduler timesteps."""

        self.timesteps = [
            torch.tensor(value, device=device) for value in range(int(steps), 0, -1)
        ]

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Return the sample unchanged for fake denoising."""

        return sample

    def step(
        self,
        *,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        eta: int,
        generator: torch.Generator,
    ) -> _FakeStepResult:
        """Return the previous sample unchanged."""

        return _FakeStepResult(sample)


class _FakeUnet:
    """UNet stand-in that returns a tensor matching the model input shape."""

    def __call__(
        self,
        *,
        x: torch.Tensor,
        mean: torch.Tensor,
        f0: torch.Tensor,
        t: torch.Tensor,
        ref,
        embed,
    ) -> torch.Tensor:
        """Produce a zero residual for fake denoising."""

        return torch.zeros_like(x)


class DiffPitcherInferenceLoopTests(unittest.TestCase):
    """Verify DiffPitcher diffusion loop behavior."""

    def test_run_diffusion_reports_each_scheduler_step(self) -> None:
        """Progress callback should receive initial and per-step diffusion messages."""

        runtime = types.SimpleNamespace(device=torch.device("cpu"), unet=_FakeUnet())
        source_mel = np.zeros((100, 4), dtype=np.float32)
        f0_bins = np.zeros(4, dtype=np.float32)
        progress_calls: list[tuple[float | None, str]] = []

        def callback(progress_value=None, text=None) -> None:
            progress_calls.append((progress_value, str(text)))

        with patch("diffusers.DDIMScheduler", _FakeScheduler):
            result = run_diffusion(
                runtime,
                source_mel,
                f0_bins,
                3,
                progress_callback=callback,
            )

        self.assertEqual((1, 100, 4), tuple(result.shape))
        self.assertEqual(4, len(progress_calls))
        self.assertEqual((0.0, "DiffPitcher diffusion: 0/3 steps complete"), progress_calls[0])
        self.assertIn("DiffPitcher diffusion step 1/3", progress_calls[1][1])
        self.assertIn("DiffPitcher diffusion step 3/3", progress_calls[-1][1])
        self.assertAlmostEqual(1.0, progress_calls[-1][0])


if __name__ == "__main__":
    unittest.main()
