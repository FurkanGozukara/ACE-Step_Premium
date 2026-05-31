"""Vendored SAM-Audio runtime import helpers."""

from __future__ import annotations

import sys
import os
from pathlib import Path


def ensure_vendor_path() -> Path:
    """Place the vendored SAM-Audio package first on ``sys.path``."""

    vendor_root = Path(__file__).resolve().parent
    vendor_root_str = str(vendor_root)
    if not sys.path or sys.path[0] != vendor_root_str:
        try:
            sys.path.remove(vendor_root_str)
        except ValueError:
            pass
        sys.path.insert(0, vendor_root_str)
    _add_local_ffmpeg_shared_bin(vendor_root)
    return vendor_root


def _add_local_ffmpeg_shared_bin(vendor_root: Path) -> None:
    """Add local shared FFmpeg DLLs to PATH when they are available."""

    project_root = vendor_root.parents[1]
    ffmpeg_bin = project_root / "tools" / "ffmpeg_shared" / "bin"
    if not ffmpeg_bin.is_dir():
        return
    ffmpeg_bin_str = str(ffmpeg_bin)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_bin_str not in path_parts:
        os.environ["PATH"] = ffmpeg_bin_str + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(ffmpeg_bin_str)
