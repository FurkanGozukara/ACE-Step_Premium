"""Tests for adaptive timestep sampling."""

from __future__ import annotations

import unittest

import torch

from acestep.training.adaptive_timestep import AdaptiveTimestepSampler


def _base_sampler(batch_size: int, device, dtype, **_kwargs):
    """Return deterministic base timesteps for sampler tests."""

    values = torch.full((batch_size,), 0.25, device=device, dtype=dtype)
    return values, values.clone()


class AdaptiveTimestepSamplerTests(unittest.TestCase):
    """Verify adaptive timestep sampling and loss updates."""

    def test_zero_ratio_uses_only_base_sampler(self) -> None:
        """A ratio of 0 should preserve the base sampler distribution."""

        sampler = AdaptiveTimestepSampler(ratio=0.0)
        t, r = sampler.sample(
            batch_size=4,
            base_sampler=_base_sampler,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(t, torch.full((4,), 0.25)))
        self.assertTrue(torch.equal(r, torch.full((4,), 0.25)))

    def test_update_tracks_high_loss_bin(self) -> None:
        """Updating with a high-loss sample should increase that bin estimate."""

        sampler = AdaptiveTimestepSampler(n_bins=4, ema_decay=0.0, ratio=1.0)
        sampler.update(
            torch.tensor([0.1, 0.8], dtype=torch.float32),
            torch.tensor([1.0, 9.0], dtype=torch.float32),
        )

        self.assertGreater(sampler._bin_loss[3].item(), sampler._bin_loss[0].item())


if __name__ == "__main__":
    unittest.main()
