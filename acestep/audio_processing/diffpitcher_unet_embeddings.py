"""Sinusoidal embeddings used by the DiffPitcher U-Net."""

from __future__ import annotations

import math

import torch
from einops import rearrange


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Return sinusoidal timestep embeddings."""

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = scale * timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(torch.nn.Module):
    """Diffusion timestep embedding module."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed integer or fractional diffusion timesteps."""

        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class PitchPosEmb(torch.nn.Module):
    """Pitch-bin positional embedding."""

    def __init__(
        self,
        dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed frame-wise pitch bins."""

        batch, length = x.shape
        flat = rearrange(x, "b l -> (b l)")
        emb = get_timestep_embedding(
            flat,
            self.dim,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return rearrange(emb, "(b l) d -> b d l", b=batch, l=length)
