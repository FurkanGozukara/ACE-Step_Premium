"""Model loading and device selection for DiffPitcher inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from .diffpitcher_bigvgan import BigVGAN
from .diffpitcher_paths import (
    DiffPitcherAssetPaths,
    default_asset_paths,
    load_json_file,
    load_yaml_file,
    require_existing_file,
)
from .diffpitcher_pitchformer import PitchFormer
from .diffpitcher_unet import UNetPitcher


@dataclass
class DiffPitcherRuntime:
    """Loaded DiffPitcher inference models for one device."""

    device: torch.device
    paths: DiffPitcherAssetPaths
    unet: UNetPitcher
    vocoder: BigVGAN
    pitchformer: PitchFormer | None


_RUNTIME_CACHE: dict[str, DiffPitcherRuntime] = {}


def get_diffpitcher_runtime(
    device: torch.device,
    *,
    needs_pitchformer: bool,
) -> DiffPitcherRuntime:
    """Load or return cached DiffPitcher models for a device."""

    key = str(device)
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        paths = default_asset_paths()
        runtime = DiffPitcherRuntime(
            device=device,
            paths=paths,
            unet=_load_unet(paths, device),
            vocoder=_load_bigvgan(paths, device),
            pitchformer=None,
        )
        _RUNTIME_CACHE[key] = runtime
    if needs_pitchformer and runtime.pitchformer is None:
        runtime.pitchformer = _load_pitchformer(runtime.paths, device)
    return runtime


def clear_diffpitcher_runtime_cache() -> None:
    """Clear cached DiffPitcher models, primarily for tests."""

    _RUNTIME_CACHE.clear()


def select_diffpitcher_device(mode: str) -> torch.device:
    """Return the device for DiffPitcher inference."""

    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DiffPitcher CUDA device selected, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_unet(paths: DiffPitcherAssetPaths, device: torch.device) -> UNetPitcher:
    """Load the DiffPitcher denoiser from safetensors."""

    config = load_yaml_file(paths.unet_config)
    unet_config = dict(config.get("unet") or {})
    unet_config["dim_mults"] = tuple(unet_config.get("dim_mults") or (1, 2, 4))
    model = UNetPitcher(**unet_config).to(device)
    state = load_file(
        str(require_existing_file(paths.unet, "U-Net checkpoint")),
        device=str(device),
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


def _load_bigvgan(paths: DiffPitcherAssetPaths, device: torch.device) -> BigVGAN:
    """Load the DiffPitcher BigVGAN generator from safetensors."""

    model = BigVGAN(load_json_file(paths.bigvgan_config)).to(device)
    state = load_file(
        str(require_existing_file(paths.bigvgan, "BigVGAN checkpoint")),
        device=str(device),
    )
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    return model.eval()


def _load_pitchformer(paths: DiffPitcherAssetPaths, device: torch.device) -> PitchFormer:
    """Load the score-based PitchFormer model from safetensors."""

    config = load_yaml_file(paths.pitchformer_config)
    model_config = dict(config.get("pitchformer") or {})
    model = PitchFormer(**model_config).to(device)
    state = load_file(
        str(require_existing_file(paths.pitchformer, "PitchFormer checkpoint")),
        device=str(device),
    )
    model.load_state_dict(state, strict=True)
    return model.eval()
