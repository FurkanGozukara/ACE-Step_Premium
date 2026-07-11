"""ACE-Step compatibility facade for the portable toolchain package.

New applications should import directly from :mod:`torch_compile_toolchain`. This
facade preserves ACE-Step's historical cache location and existing import paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

from torch_compile_toolchain.environment import (
    CompileToolchainStatus,
    compile_environment_report as _portable_report,
    ensure_compile_environment as _portable_ensure,
    prepare_compile_subprocess_env as _portable_subprocess_env,
)


def ensure_compile_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
) -> CompileToolchainStatus:
    """Prepare ACE-Step while delegating discovery to the portable package."""

    target_env = os.environ if env is None else env
    return _portable_ensure(
        target_env,
        project_root=project_root,
        cache_dir=cache_dir or _legacy_cache_dir(target_env, project_root),
        require_cuda_toolkit=require_cuda_toolkit,
        require_ninja=require_ninja,
    )


def prepare_compile_subprocess_env(
    env: MutableMapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    compile_requested: bool = True,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
) -> dict[str, str]:
    """Prepare an ACE-Step child process with the portable discovery package."""

    source_env = os.environ if env is None else env
    return _portable_subprocess_env(
        source_env,
        project_root=project_root,
        cache_dir=cache_dir or _legacy_cache_dir(source_env, project_root),
        compile_requested=compile_requested,
        require_cuda_toolkit=require_cuda_toolkit,
        require_ninja=require_ninja,
    )


def compile_environment_report(
    *,
    project_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_cuda_toolkit: bool = False,
    require_ninja: bool = False,
) -> CompileToolchainStatus:
    """Return ACE-Step's current-process toolchain report."""

    return _portable_report(
        project_root=project_root,
        cache_dir=cache_dir or _legacy_cache_dir(os.environ, project_root),
        require_cuda_toolkit=require_cuda_toolkit,
        require_ninja=require_ninja,
    )


def _legacy_cache_dir(
    env: MutableMapping[str, str],
    project_root: str | Path | None,
) -> Path:
    root_text = str(project_root or env.get("ACESTEP_PROJECT_ROOT") or "").strip()
    root = Path(root_text).expanduser().resolve() if root_text else Path.cwd()
    return root / ".cache" / "acestep" / "torch_compile"


__all__ = [
    "CompileToolchainStatus",
    "compile_environment_report",
    "ensure_compile_environment",
    "prepare_compile_subprocess_env",
]
