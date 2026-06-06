"""Filesystem paths for local DiffPitcher assets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DiffPitcherAssetPaths:
    """Resolved local DiffPitcher checkpoint and config paths."""

    unet: Path
    pitchformer: Path
    bigvgan: Path
    unet_config: Path
    pitchformer_config: Path
    bigvgan_config: Path


def project_root() -> Path:
    """Return the active ACE-Step project root."""

    env_root = os.environ.get("ACESTEP_PROJECT_ROOT")
    return Path(env_root).resolve() if env_root else Path(__file__).resolve().parents[2]


def default_asset_paths(root: Path | None = None) -> DiffPitcherAssetPaths:
    """Return the default local DiffPitcher asset paths."""

    resolved_root = (root or project_root()).resolve()
    models_dir = resolved_root / "models"
    config_dir = resolved_root / "Diff-Pitcher_configs"
    return DiffPitcherAssetPaths(
        unet=models_dir / "Diff-Pitcher_world_fixed_40.safetensors",
        pitchformer=models_dir / "Diff-Pitcher_transformer_pitch_360.safetensors",
        bigvgan=models_dir / "Diff-Pitcher_bigvgan_24khz_100band.safetensors",
        unet_config=config_dir / "DiffWorld_24k.yaml",
        pitchformer_config=config_dir / "Pitchformer.yaml",
        bigvgan_config=config_dir / "bigvgan_24khz_100band_config.json",
    )


def require_existing_file(path: Path, label: str) -> Path:
    """Return an existing file path or raise a clear asset error."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing DiffPitcher {label}: {path}")
    return path


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    require_existing_file(path, "YAML config")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON config file into a dictionary."""

    require_existing_file(path, "JSON config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
