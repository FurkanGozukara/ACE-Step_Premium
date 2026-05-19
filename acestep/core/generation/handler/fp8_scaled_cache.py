"""Disk cache helpers for scaled FP8 DiT weights."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger


CACHE_VERSION = 1


def load_cached_entries(cache_file: Path) -> dict[str, dict[str, torch.Tensor]] | None:
    """Load cached FP8 entries when the cache version matches."""

    if not cache_file.exists():
        return None
    try:
        payload = torch.load(cache_file, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("[fp8_scaled] Ignoring unreadable cache {}: {}", cache_file, exc)
        return None

    if payload.get("version") != CACHE_VERSION or not isinstance(payload.get("entries"), dict):
        return None
    logger.info("[fp8_scaled] Reusing cache {}", cache_file)
    return payload["entries"]


def save_cached_entries(
    cache_file: Path,
    entries: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Persist FP8 entries atomically."""

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    torch.save({"version": CACHE_VERSION, "entries": entries}, tmp_file)
    os.replace(tmp_file, cache_file)
    logger.info("[fp8_scaled] Wrote cache {}", cache_file)


def cache_file_for_checkpoint(
    checkpoint_path: str | os.PathLike[str] | None,
    *,
    variant: str = "",
) -> Path:
    """Return the cache file path for a checkpoint fingerprint."""

    fingerprint = _checkpoint_fingerprint(checkpoint_path)
    if variant:
        fingerprint = f"{fingerprint}|variant:{variant}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    default_root = Path.cwd() / ".cache" / "acestep" / "fp8_scaled"
    cache_root = Path(os.environ.get("ACESTEP_FP8_CACHE_DIR", "") or default_root)
    return cache_root / f"{digest[:24]}.pt"


def _checkpoint_fingerprint(checkpoint_path: str | os.PathLike[str] | None) -> str:
    """Build a fingerprint from checkpoint file names, sizes, and mtimes."""

    if checkpoint_path is None:
        return "unknown"
    root = Path(checkpoint_path).expanduser()
    if not root.exists():
        return str(root)

    parts: list[str] = [str(root.resolve())]
    suffixes = {".safetensors", ".bin", ".json", ".pt"}
    candidates = [path for path in root.rglob("*") if path.suffix.lower() in suffixes]
    for path in sorted(candidates):
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        parts.append(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)
