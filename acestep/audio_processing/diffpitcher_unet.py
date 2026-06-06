"""DiffPitcher U-Net denoiser used for pitch-conditioned mel generation."""

from __future__ import annotations

import torch

from .diffpitcher_unet_blocks import (
    Block,
    Downsample,
    LinearAttention,
    Mish,
    Residual,
    ResnetBlock,
    Rezero,
    Upsample,
)
from .diffpitcher_unet_embeddings import PitchPosEmb, Timesteps


class UNetPitcher(torch.nn.Module):
    """Pitch-conditioned diffusion U-Net matching the DiffPitcher checkpoint."""

    def __init__(
        self,
        dim_base: int,
        dim_cond: int,
        use_ref_t: bool,
        use_embed: bool,
        dim_embed: int | None = 256,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        pitch_type: str = "bins",
    ) -> None:
        super().__init__()
        if use_ref_t or use_embed:
            raise ValueError("This inference build supports DiffPitcher's non-timbre config.")
        _ = (dim_embed, pitch_type)
        dim_in = 2
        self.use_ref_t = use_ref_t
        self.use_embed = use_embed
        self.pitch_type = pitch_type
        self.time_pos_emb = Timesteps(
            num_channels=dim_base,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_base, dim_base * 4),
            Mish(),
            torch.nn.Linear(dim_base * 4, dim_base),
        )
        self.pitch_pos_emb = PitchPosEmb(dim_cond)
        self.pitch_mlp = torch.nn.Sequential(
            torch.nn.Conv1d(dim_cond, dim_cond * 4, 1, stride=1),
            Mish(),
            torch.nn.Conv1d(dim_cond * 4, dim_cond, 1, stride=1),
        )
        dim_in += dim_cond
        dims = [dim_in, *[dim_base * multiplier for multiplier in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(in_dim, out_dim, time_emb_dim=dim_base),
                ResnetBlock(out_dim, out_dim, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(out_dim))),
                Downsample(out_dim) if not is_last else torch.nn.Identity(),
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        for in_dim, out_dim in reversed(in_out[1:]):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(out_dim * 2, in_dim, time_emb_dim=dim_base),
                ResnetBlock(in_dim, in_dim, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(in_dim))),
                Upsample(in_dim),
            ]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        f0: torch.Tensor,
        t: torch.Tensor | int,
        ref: torch.Tensor | None = None,
        embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict diffusion noise from source mel, target pitch bins, and timestep."""

        _ = (ref, embed)
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        if len(t.shape) == 0:
            t = t * torch.ones(x.shape[0], dtype=t.dtype, device=x.device)
        timestep = self.mlp(self.time_pos_emb(t))
        stacked = torch.stack([x, mean], 1)
        pitch = self.pitch_pos_emb(f0)
        pitch = self.pitch_mlp(pitch).unsqueeze(2)
        pitch = torch.cat(stacked.shape[2] * [pitch], 2)
        hidden = torch.cat([stacked, pitch], 1)
        hiddens = []
        for resnet1, resnet2, attn, downsample in self.downs:
            hidden = resnet1(hidden, timestep)
            hidden = resnet2(hidden, timestep)
            hidden = attn(hidden)
            hiddens.append(hidden)
            hidden = downsample(hidden)
        hidden = self.mid_block1(hidden, timestep)
        hidden = self.mid_attn(hidden)
        hidden = self.mid_block2(hidden, timestep)
        for resnet1, resnet2, attn, upsample in self.ups:
            hidden = torch.cat((hidden, hiddens.pop()), dim=1)
            hidden = resnet1(hidden, timestep)
            hidden = resnet2(hidden, timestep)
            hidden = attn(hidden)
            hidden = upsample(hidden)
        return self.final_conv(self.final_block(hidden)).squeeze(1)
