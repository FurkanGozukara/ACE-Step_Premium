"""Output-directory resolution helpers for generated audio artifacts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs"
_ACTIVE_RESULTS_DIR: ContextVar[Path | None] = ContextVar(
    "acestep_active_results_dir",
    default=None,
)


def _resolve_configured_results_dir() -> Path:
    """Return the environment-configured results directory when present."""

    configured = os.environ.get("ACESTEP_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_RESULTS_DIR


def get_results_dir() -> Path:
    """Return the active output root directory, creating it if needed."""

    target = _ACTIVE_RESULTS_DIR.get() or _resolve_configured_results_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_active_results_dir() -> Path | None:
    """Return the current scoped output directory override, if one is active."""

    return _ACTIVE_RESULTS_DIR.get()


@contextmanager
def use_results_dir(output_dir: str | Path) -> Iterator[Path]:
    """Temporarily route generated artifacts into ``output_dir``."""

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    token = _ACTIVE_RESULTS_DIR.set(target)
    try:
        yield target
    finally:
        _ACTIVE_RESULTS_DIR.reset(token)


def create_generation_run_dir() -> Path:
    """Allocate the next sequential numbered generation folder."""

    root = get_results_dir()
    max_index = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            try:
                max_index = max(max_index, int(child.name))
            except ValueError:
                continue

    run_dir = root / f"{max_index + 1:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
