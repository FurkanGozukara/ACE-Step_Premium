"""Adaptive timestep sampling for LoRA/DoRA flow-matching training."""

from __future__ import annotations

from collections.abc import Callable

import torch


class AdaptiveTimestepSampler:
    """Mix standard timestep samples with loss-weighted timestep bins.

    Args:
        n_bins: Number of uniform bins over the ``[0, 1]`` timestep range.
        ema_decay: EMA smoothing for per-bin loss estimates.
        ratio: Fraction of each batch sampled from adaptive bins.
    """

    def __init__(
        self,
        n_bins: int = 10,
        ema_decay: float = 0.99,
        ratio: float = 0.0,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        self.n_bins = n_bins
        self.ema_decay = ema_decay
        self.ratio = ratio
        self._bin_loss = torch.ones(n_bins, dtype=torch.float32)

    @torch.no_grad()
    def update(self, timesteps: torch.Tensor, losses: torch.Tensor) -> None:
        """Update loss estimates for sampled timestep bins."""

        bins = timesteps.detach().float().cpu().mul(self.n_bins).long()
        bins = bins.clamp(0, self.n_bins - 1)
        sample_losses = losses.detach().float().cpu()
        decay = self.ema_decay

        for bin_idx in range(self.n_bins):
            mask = bins == bin_idx
            if mask.any():
                bin_mean = sample_losses[mask].mean()
                self._bin_loss[bin_idx] = decay * self._bin_loss[bin_idx] + (
                    1.0 - decay
                ) * bin_mean

    def sample(
        self,
        *,
        batch_size: int,
        base_sampler: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
        **base_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(t, r)`` using adaptive bins plus the base sampler."""

        adaptive_count = int(batch_size * self.ratio)
        base_count = batch_size - adaptive_count
        parts_t: list[torch.Tensor] = []
        parts_r: list[torch.Tensor] = []

        if base_count > 0:
            t_base, r_base = base_sampler(
                batch_size=base_count,
                device=device,
                dtype=dtype,
                **base_kwargs,
            )
            parts_t.append(t_base)
            parts_r.append(r_base)

        if adaptive_count > 0:
            weights = self._bin_loss.clamp(min=1e-8)
            probabilities = weights / weights.sum()
            bin_indices = torch.multinomial(
                probabilities,
                adaptive_count,
                replacement=True,
            )
            bin_low = bin_indices.float() / self.n_bins
            bin_high = (bin_indices.float() + 1.0) / self.n_bins
            t_adaptive = bin_low + (bin_high - bin_low) * torch.rand(adaptive_count)
            t_adaptive = t_adaptive.to(device=device, dtype=dtype)
            parts_t.append(t_adaptive)
            parts_r.append(t_adaptive.clone())

        t = torch.cat(parts_t, dim=0)
        r = torch.cat(parts_r, dim=0)
        permutation = torch.randperm(batch_size, device=device)
        return t[permutation], r[permutation]
