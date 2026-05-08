"""Scaled FP8 quantization helpers for DiT linear weights.

The implementation follows the musubi-tuner approach: FP8 weights are stored
with per-block/per-channel scale tensors, and affected ``nn.Linear`` modules
dequantize weights inside a patched forward pass. Quantized weights are cached
per checkpoint fingerprint so repeated launches can reuse the scaled tensors.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .fp8_scaled_cache import (
    cache_file_for_checkpoint,
    load_cached_entries,
    save_cached_entries,
)

_BLOCK_SIZE = 64


def apply_fp8_scaled_quantization(
    model: nn.Module,
    *,
    checkpoint_path: str | os.PathLike[str] | None,
    device: str | torch.device,
) -> None:
    """Apply cached scaled-FP8 quantization to decoder linear weights.

    Args:
        model: Loaded ACE-Step DiT model to modify in-place.
        checkpoint_path: Source checkpoint directory used to fingerprint cache.
        device: Runtime device used for quantization calculations.
    """

    _ensure_float8_available()
    entries = _load_or_build_cache(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    patched = _apply_entries_to_model(model, entries)
    logger.info("[fp8_scaled] Applied scaled FP8 to {} Linear layers", patched)


def _ensure_float8_available() -> None:
    """Raise when the installed PyTorch lacks required FP8 dtypes."""

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Scaled FP8 requires a PyTorch build with float8_e4m3fn support.")


def _load_or_build_cache(
    *,
    model: nn.Module,
    checkpoint_path: str | os.PathLike[str] | None,
    device: str | torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load a compatible FP8 cache or build and persist one."""

    cache_file = _cache_file_for_checkpoint(checkpoint_path)
    cached_entries = load_cached_entries(cache_file)
    if cached_entries is not None:
        return cached_entries

    entries = _build_fp8_entries(model, device)
    save_cached_entries(cache_file, entries)
    return entries


def _build_fp8_entries(
    model: nn.Module,
    device: str | torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Quantize target linear weights and return CPU cache entries."""

    calc_device = _resolve_calc_device(device)
    entries: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in _iter_target_linears(model):
        weight = module.weight.detach().to(calc_device)
        quantized, scale = _quantize_weight(name, weight)
        entries[name] = {
            "weight": quantized.cpu(),
            "scale_weight": scale.to(module.weight.dtype).cpu(),
        }
        if calc_device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info("[fp8_scaled] Quantized {} Linear layers", len(entries))
    return entries


def _resolve_calc_device(device: str | torch.device) -> torch.device:
    """Return the calculation device, preferring CUDA when requested."""

    requested = torch.device(device if str(device) != "auto" else "cuda")
    if requested.type == "cuda" and torch.cuda.is_available():
        return requested
    return torch.device("cpu")


def _iter_target_linears(model: nn.Module):
    """Yield decoder-side linear modules that are safe to quantize."""

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        if not parts or parts[0] != "decoder":
            continue
        if any(part in {"tokenizer", "detokenizer"} for part in parts):
            continue
        if module.weight.ndim != 2:
            continue
        yield name, module


def _quantize_weight(name: str, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one linear weight using block FP8 scales."""

    max_value = torch.finfo(torch.float8_e4m3fn).max
    min_value = -max_value
    original_shape = weight.shape
    mode = "block"
    if weight.shape[1] % _BLOCK_SIZE != 0:
        mode = "channel"
        logger.debug("[fp8_scaled] {} uses channel scale fallback", name)

    if mode == "block":
        out_features, in_features = weight.shape
        weight = weight.contiguous().view(out_features, in_features // _BLOCK_SIZE, _BLOCK_SIZE)
        scale = torch.max(torch.abs(weight), dim=2, keepdim=True).values
    else:
        scale = torch.max(torch.abs(weight), dim=1, keepdim=True).values

    scale = torch.clamp(scale / max_value, min=1e-8).to(torch.float32)
    quantized = torch.div(weight.to(torch.float32), scale).nan_to_num_(0.0)
    quantized = quantized.clamp_(min=min_value, max=max_value).to(torch.float8_e4m3fn)
    if mode == "block":
        quantized = quantized.view(original_shape)
    return quantized, scale


def _apply_entries_to_model(model: nn.Module, entries: dict[str, dict[str, torch.Tensor]]) -> int:
    """Install FP8 weights, scale buffers, and patched forwards."""

    patched = 0
    modules = dict(model.named_modules())
    for name, entry in entries.items():
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            continue
        weight = entry.get("weight")
        scale = entry.get("scale_weight")
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            continue

        target_device = module.weight.device
        module.weight = nn.Parameter(weight.to(target_device), requires_grad=False)
        module.register_buffer("scale_weight", scale.to(target_device), persistent=True)
        module.forward = _fp8_linear_forward.__get__(module, type(module))
        patched += 1
    return patched


def _fp8_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Forward for FP8 scaled linears that dequantizes weights on demand."""

    scale = self.scale_weight.to(device=self.weight.device)
    original_dtype = scale.dtype
    if scale.ndim < 3:
        dequantized_weight = self.weight.to(original_dtype) * scale
    else:
        out_features, num_blocks, _ = scale.shape
        dequantized_weight = self.weight.to(original_dtype).contiguous()
        dequantized_weight = dequantized_weight.view(out_features, num_blocks, -1)
        dequantized_weight = (dequantized_weight * scale).view(self.weight.shape)
    return F.linear(x, dequantized_weight, self.bias)


def _cache_file_for_checkpoint(checkpoint_path: str | os.PathLike[str] | None) -> Path:
    """Return the cache file path for a checkpoint fingerprint."""

    return cache_file_for_checkpoint(checkpoint_path)
