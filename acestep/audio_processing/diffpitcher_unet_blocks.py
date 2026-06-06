"""U-Net building blocks used by the DiffPitcher denoiser."""

from __future__ import annotations

import torch
from einops import rearrange


class Mish(torch.nn.Module):
    """Mish activation matching the DiffPitcher checkpoint graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Mish nonlinearity."""

        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(torch.nn.Module):
    """Transposed-convolution upsampling block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the feature map by 2x."""

        return self.conv(x)


class Downsample(torch.nn.Module):
    """Strided-convolution downsampling block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the feature map by 2x."""

        return self.conv(x)


class Rezero(torch.nn.Module):
    """Residual gate initialized at zero."""

    def __init__(self, fn: torch.nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped module through a learned scalar gate."""

        return self.fn(x) * self.g


class Block(torch.nn.Module):
    """Convolution, group norm, and Mish block."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one convolutional block."""

        return self.block(x)


class ResnetBlock(torch.nn.Module):
    """ResNet block conditioned by the diffusion timestep embedding."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Run a residual block with timestep conditioning."""

        hidden = self.block1(x)
        hidden += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        hidden = self.block2(hidden)
        return hidden + self.res_conv(x)


class LinearAttention(torch.nn.Module):
    """Linearized 2D attention used in the denoiser."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        q_norm: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)
        self.q_norm = q_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear attention to a 2D feature map."""

        _, _, height, width = x.shape
        query, key, value = rearrange(
            self.to_qkv(x),
            "b (qkv heads c) h w -> qkv b heads c (h w)",
            heads=self.heads,
            qkv=3,
        )
        key = key.softmax(dim=-1)
        if self.q_norm:
            query = query.softmax(dim=-2)
        context = torch.einsum("bhdn,bhen->bhde", key, value)
        out = torch.einsum("bhde,bhdn->bhen", context, query)
        out = rearrange(
            out,
            "b heads c (h w) -> b (heads c) h w",
            heads=self.heads,
            h=height,
            w=width,
        )
        return self.to_out(out)


class Residual(torch.nn.Module):
    """Residual wrapper for attention blocks."""

    def __init__(self, fn: torch.nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return wrapped output plus input."""

        return self.fn(x) + x
