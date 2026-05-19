"""Safe-root payload helpers for isolated training workers."""

from __future__ import annotations

import os
from typing import Any, Iterable

from loguru import logger

from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import get_safe_roots, set_safe_roots


def build_worker_safe_roots(
    project_root: object,
    *paths: object,
    samples: Iterable[Any] = (),
) -> list[str]:
    """Build the filesystem roots a worker needs for one dataset action.

    Args:
        project_root: Repository root used as the worker working directory.
        *paths: Request/result/output paths that should remain writable.
        samples: Dataset samples whose source-audio directories should be valid.

    Returns:
        Existing canonical directories suitable for ``set_safe_roots``.
    """

    candidates: list[object] = [project_root, *get_safe_roots(), *paths]
    candidates.extend(getattr(sample, "audio_path", "") for sample in samples)
    return _dedupe_existing_roots(candidates)


def apply_worker_safe_roots(payload: dict[str, Any]) -> None:
    """Apply serialized safe roots before a worker reads or writes dataset files."""

    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    candidates: list[object] = [
        *(payload.get("safe_roots") or []),
        payload.get("project_root"),
        payload.get("dataset_path"),
        payload.get("result_dataset_path"),
        payload.get("output_dir"),
        settings.get("save_path"),
    ]
    roots = _dedupe_existing_roots(candidates)
    if not roots:
        logger.warning("Training worker received no usable safe roots; keeping defaults")
        return
    set_safe_roots(roots)


def _dedupe_existing_roots(paths: Iterable[object]) -> list[str]:
    """Return existing canonical directories while preserving order."""

    roots: list[str] = []
    seen: set[str] = set()
    for path in paths:
        root = _nearest_existing_directory(path)
        if root is None:
            continue
        key = os.path.normcase(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _nearest_existing_directory(path: object) -> str | None:
    """Return the nearest existing directory for a file or directory path."""

    raw_path = normalize_user_path(path)
    if not raw_path:
        return None

    try:
        candidate = os.path.normpath(os.path.realpath(os.path.abspath(raw_path)))
    except (OSError, ValueError) as exc:
        logger.debug(f"Skipping worker safe-root path {raw_path!r}: {exc}")
        return None

    while candidate and not os.path.isdir(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent

    return candidate if candidate else None
