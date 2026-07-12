from collections.abc import Mapping
from functools import lru_cache
from numbers import Real

import torch
from torch import nn

from nanovllm.utils.compat import maybe_compile


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve unindexed CUDA devices so cached modules cannot cross GPUs."""
    resolved = torch.device(device) if device is not None else torch.get_default_device()
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        resolved_device = _resolve_device(device)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=torch.float,
                    device=resolved_device,
                )
                / rotary_dim
            )
        )
        t = torch.arange(
            max_position_embeddings,
            dtype=torch.float,
            device=resolved_device,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @maybe_compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _resolve_rope_base(
    base: float,
    rope_scaling: Mapping[str, object] | None,
) -> float:
    """Normalize the default RoPE schema used by supported Transformers releases."""
    if rope_scaling is None:
        resolved_base = base
    else:
        if not isinstance(rope_scaling, Mapping):
            raise TypeError(
                "rope_scaling must be a mapping or None, "
                f"got {type(rope_scaling).__name__}"
            )

        rope_type = rope_scaling.get("rope_type")
        legacy_rope_type = rope_scaling.get("type")
        if rope_type is not None and legacy_rope_type is not None and rope_type != legacy_rope_type:
            raise ValueError(
                "Conflicting RoPE types: "
                f"rope_type={rope_type!r}, type={legacy_rope_type!r}"
            )
        rope_type = rope_type if rope_type is not None else legacy_rope_type
        if rope_type is None:
            extra_keys = set(rope_scaling) - {
                "partial_rotary_factor",
                "rope_theta",
                "rope_type",
                "type",
            }
            if extra_keys:
                keys = ", ".join(sorted(str(key) for key in extra_keys))
                raise ValueError(
                    "nano-vLLM cannot infer the RoPE type for scaling fields: "
                    f"{keys}"
                )
            rope_type = "default"
        if rope_type != "default":
            raise ValueError(
                "nano-vLLM only supports default RoPE, "
                f"got rope_type={rope_type!r}"
            )

        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0)
        if partial_rotary_factor != 1.0:
            raise ValueError(
                "nano-vLLM only supports full-head rotary embeddings, "
                f"got partial_rotary_factor={partial_rotary_factor!r}"
            )
        resolved_base = rope_scaling.get("rope_theta", base)

    if isinstance(resolved_base, bool) or not isinstance(resolved_base, Real):
        raise TypeError(
            "RoPE theta must be a real number, "
            f"got {type(resolved_base).__name__}"
        )
    if resolved_base <= 0:
        raise ValueError(f"RoPE theta must be positive, got {resolved_base!r}")
    return float(resolved_base)


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    device: str,
) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        device=torch.device(device),
    )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Mapping[str, object] | None = None,
    device: torch.device | str | None = None,
) -> RotaryEmbedding:
    """Return a cached rotary embedding after normalizing RoPE configuration."""
    resolved_base = _resolve_rope_base(base, rope_scaling)
    resolved_device = _resolve_device(device)
    cache_args = (
        head_size,
        rotary_dim,
        max_position,
        resolved_base,
        str(resolved_device),
    )
    rotary_emb = _get_rope_cached(*cache_args)
    if rotary_emb.cos_sin_cache.device != resolved_device:
        _get_rope_cached.cache_clear()
        rotary_emb = _get_rope_cached(*cache_args)
    return rotary_emb
