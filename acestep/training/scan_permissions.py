"""Filesystem roots allowed for dataset scanning and Gradio file serving."""

from __future__ import annotations

import ctypes
import os
import string
from typing import Iterable

from loguru import logger

from acestep.training.path_safety import add_safe_roots

_WINDOWS_READABLE_DRIVE_TYPES = {2, 3, 4, 5, 6}


def _normalise_existing_directory(path: str) -> str | None:
    """Return a canonical directory path if it is readable enough to inspect."""
    try:
        candidate = os.path.normpath(os.path.realpath(os.path.abspath(path)))
        if os.path.isdir(candidate):
            return candidate
    except (OSError, ValueError) as exc:
        logger.debug(f"Skipping scan permission path {path!r}: {exc}")
    return None


def _dedupe_existing_directories(paths: Iterable[str]) -> list[str]:
    """Normalise, filter, and deduplicate existing directories."""
    roots: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalised = _normalise_existing_directory(path)
        if normalised is None:
            continue
        key = os.path.normcase(normalised)
        if key in seen:
            continue
        seen.add(key)
        roots.append(normalised)
    return roots


def _windows_drive_roots() -> list[str]:
    """Return existing Windows drive roots without walking their contents."""
    try:
        kernel32 = ctypes.windll.kernel32
        drive_mask = kernel32.GetLogicalDrives()
    except (AttributeError, OSError):
        drive_mask = 0

    candidate_roots: list[str] = []
    if drive_mask:
        for index, letter in enumerate(string.ascii_uppercase):
            if drive_mask & (1 << index):
                candidate_roots.append(f"{letter}:\\")
    else:
        candidate_roots = [f"{letter}:\\" for letter in string.ascii_uppercase]

    roots: list[str] = []
    for root in candidate_roots:
        try:
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(root)
        except (AttributeError, OSError):
            drive_type = 0
        if drive_type and drive_type not in _WINDOWS_READABLE_DRIVE_TYPES:
            continue
        roots.append(root)
    return roots


def _environment_scan_roots() -> list[str]:
    """Return operator-provided scan roots from ACESTEP_SCAN_ALLOWED_ROOTS."""
    raw_value = os.environ.get("ACESTEP_SCAN_ALLOWED_ROOTS", "")
    if not raw_value.strip():
        return []
    return [part.strip() for part in raw_value.split(os.pathsep) if part.strip()]


def discover_local_filesystem_roots() -> list[str]:
    """Discover broad local roots that can contain user audio datasets."""
    if os.name == "nt":
        roots = _windows_drive_roots()
    else:
        roots = [os.path.abspath(os.sep)]
    discovered = _dedupe_existing_directories(roots)
    if discovered:
        return discovered
    fallback = _normalise_existing_directory(os.getcwd())
    return [fallback] if fallback else []


def configure_scan_permissions(extra_paths: Iterable[str] = ()) -> list[str]:
    """Allow dataset scanning under discovered roots and explicit app paths.

    Args:
        extra_paths: Additional directories to allow, usually Gradio output and
            explicit ``--allowed-path`` values.

    Returns:
        Canonical directories added to the training safe-root allowlist.
    """
    roots = [
        *discover_local_filesystem_roots(),
        *_environment_scan_roots(),
        *extra_paths,
    ]
    allowed_roots = _dedupe_existing_directories(roots)
    if allowed_roots:
        add_safe_roots(allowed_roots)
    return allowed_roots
