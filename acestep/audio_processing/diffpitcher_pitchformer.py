"""PitchFormer model used by DiffPitcher score-based pitch inference."""

from __future__ import annotations

import math

import torch
from einops import rearrange


class LinearAttention(torch.nn.Module):
    """Linear 1D attention used by PitchFormer."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        q_norm: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv1d(hidden_dim, dim, 1)
        self.q_norm = q_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear attention to frame features."""

        x = x.permute(0, 2, 1)
        query, key, value = rearrange(
            self.to_qkv(x),
            "b (qkv heads c) l -> qkv b heads c l",
            heads=self.heads,
            qkv=3,
        )
        key = key.softmax(dim=-1)
        if self.q_norm:
            query = query.softmax(dim=-2)
        context = torch.einsum("bhdn,bhen->bhde", key, value)
        out = torch.einsum("bhde,bhdn->bhen", context, query)
        out = rearrange(out, "b heads c l -> b (heads c) l", heads=self.heads)
        return self.to_out(out).permute(0, 2, 1)


class TransformerBlock(torch.nn.Module):
    """PitchFormer transformer block."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attention = LinearAttention(dim, heads=n_heads, dim_head=dim // n_heads)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
        )
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and feed-forward residual paths."""

        hidden = self.norm1(x)
        x = x + self.dropout1(self.attention(hidden))
        hidden = self.norm2(x)
        return x + self.dropout2(self.feed_forward(hidden))


class PitchFormer(torch.nn.Module):
    """Predict sung F0 from source mel features and MIDI pitch guidance."""

    def __init__(self, n_mels: int, hidden_size: int, attn_layers: int = 4) -> None:
        super().__init__()
        self.sp_linear = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, hidden_size, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1),
        )
        self.midi_linear = torch.nn.Sequential(
            torch.nn.Conv1d(1, hidden_size, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1),
        )
        self.hidden_size = hidden_size
        self.pos_conv = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=63, padding=31)
        std = math.sqrt(4.0 / (63 * hidden_size))
        torch.nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        torch.nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = torch.nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = torch.nn.Sequential(self.pos_conv, torch.nn.SiLU())
        self.attn_block = torch.nn.ModuleList(
            [TransformerBlock(hidden_size, 4) for _ in range(attn_layers)]
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, midi: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """Predict frame-wise F0 in Hz."""

        midi_hidden = self.midi_linear(midi.unsqueeze(1))
        spectral_hidden = self.sp_linear(sp)
        hidden = torch.cat([midi_hidden, spectral_hidden], dim=1)
        hidden = hidden + self.pos_conv(hidden)
        hidden = hidden.permute(0, 2, 1)
        for layer in self.attn_block:
            hidden = layer(hidden)
        return self.linear(hidden).squeeze(-1)
