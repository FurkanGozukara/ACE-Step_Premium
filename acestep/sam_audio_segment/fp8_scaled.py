"""Scaled FP8 cache for SAM-Audio linear weights."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from acestep.core.generation.handler.fp8_scaled_cache import (
    cache_file_for_checkpoint,
    load_cached_entries,
    save_cached_entries,
)

_BLOCK_SIZE = 64
_TARGET_PREFIXES = ("transformer", "proj", "memory_proj")


def apply_sam_fp8_scaled(
    model: nn.Module,
    *,
    checkpoint_path: str | os.PathLike[str] | None,
    device: str | torch.device,
) -> None:
    """Apply cached scaled FP8 to SAM-Audio transformer linears."""

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Scaled FP8 requires PyTorch float8_e4m3fn support.")
    cache_file = _cache_file(checkpoint_path)
    entries = load_cached_entries(cache_file)
    if entries is None:
        entries = _build_entries(model, device)
        save_cached_entries(cache_file, entries)
    patched = _apply_entries(model, entries)
    logger.info("[sam_audio_fp8] Applied scaled FP8 to {} Linear layers", patched)


def _build_entries(
    model: nn.Module,
    device: str | torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build CPU FP8 cache entries for target linears."""

    calc_device = _resolve_calc_device(device)
    entries: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in _iter_target_linears(model):
        weight = module.weight.detach().to(calc_device)
        quantized, scale = _quantize_weight(weight)
        entries[name] = {
            "weight": quantized.cpu(),
            "scale_weight": scale.to(module.weight.dtype).cpu(),
        }
        if calc_device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info("[sam_audio_fp8] Quantized {} Linear layers", len(entries))
    return entries


def _iter_target_linears(model: nn.Module):
    """Yield SAM-Audio Linear modules that carry most model VRAM."""

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module.weight.ndim != 2:
            continue
        if name.startswith(_TARGET_PREFIXES):
            yield name, module


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one Linear weight with block or channel scales."""

    max_value = torch.finfo(torch.float8_e4m3fn).max
    min_value = -max_value
    original_shape = weight.shape
    if weight.shape[1] % _BLOCK_SIZE == 0:
        out_features, in_features = weight.shape
        reshaped = weight.contiguous().view(out_features, in_features // _BLOCK_SIZE, _BLOCK_SIZE)
        scale = torch.max(torch.abs(reshaped), dim=2, keepdim=True).values
        quantized = torch.div(reshaped.float(), torch.clamp(scale / max_value, min=1e-8))
        scale = torch.clamp(scale / max_value, min=1e-8).float()
        return quantized.nan_to_num_(0.0).clamp_(min_value, max_value).to(
            torch.float8_e4m3fn
        ).view(original_shape), scale
    scale = torch.max(torch.abs(weight), dim=1, keepdim=True).values
    scale = torch.clamp(scale / max_value, min=1e-8).float()
    quantized = torch.div(weight.float(), scale).nan_to_num_(0.0)
    return quantized.clamp_(min_value, max_value).to(torch.float8_e4m3fn), scale


def _apply_entries(model: nn.Module, entries: dict[str, dict[str, torch.Tensor]]) -> int:
    """Install FP8 weights and patched forwards."""

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
    """Dequantize a scaled FP8 Linear weight for one forward call."""

    scale = self.scale_weight.to(device=self.weight.device)
    original_dtype = scale.dtype
    if scale.ndim < 3:
        weight = self.weight.to(original_dtype) * scale
    else:
        out_features, num_blocks, _ = scale.shape
        weight = self.weight.to(original_dtype).contiguous()
        weight = weight.view(out_features, num_blocks, -1)
        weight = (weight * scale).view(self.weight.shape)
    return F.linear(x, weight, self.bias)


def _resolve_calc_device(device: str | torch.device) -> torch.device:
    """Return CUDA for auto when available, otherwise CPU."""

    requested = torch.device(device if str(device) != "auto" else "cuda")
    if requested.type == "cuda" and torch.cuda.is_available():
        return requested
    return torch.device("cpu")


def _cache_file(checkpoint_path: str | os.PathLike[str] | None) -> Path:
    """Return a SAM-specific FP8 cache file path."""

    return cache_file_for_checkpoint(checkpoint_path, variant="sam_audio")
