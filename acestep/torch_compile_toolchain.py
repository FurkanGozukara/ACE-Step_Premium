"""Toolchain discovery helpers for optional ``torch.compile``."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

from loguru import logger

from acestep.torch_compile_compiler_probe import probe_cuda_compiler
from acestep.torch_compile_discovery import (
    discover_cuda_environment,
    discover_ninja,
    discover_posix_compiler,
)
from acestep.torch_compile_msvc import (
    describe_msvc_environment,
    has_cl_exe,
    load_msvc_environment,
)


@dataclass(frozen=True)
class CompileToolchainStatus:
    """Summarize the runtime environment prepared for ``torch.compile``."""

    ok: bool
    detail: str
    changed: bool = False


_CACHED_ENV_DELTA: dict[str, str] | None = None
_CACHED_STATUS: CompileToolchainStatus | None = None


def ensure_compile_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
) -> CompileToolchainStatus:
    """Prepare the current or supplied environment for PyTorch Inductor.

    Args:
        env: Environment mapping to update. Defaults to ``os.environ``.
        project_root: Optional project root used for stable compile cache folders.

    Returns:
        Toolchain status with a human-readable detail string.
    """

    target_env = os.environ if env is None else env
    _ensure_compile_cache_dirs(target_env, project_root)
    cuda_status = discover_cuda_environment(target_env)
    ninja_status = discover_ninja(target_env)
    if sys.platform != "win32":
        compiler_status = discover_posix_compiler(target_env)
        detail = _join_details(cuda_status.detail, compiler_status.detail, ninja_status.detail)
        return CompileToolchainStatus(compiler_status.ok, detail, cuda_status.changed)

    if has_cl_exe(target_env):
        probe = probe_cuda_compiler(target_env, platform_name=sys.platform)
        if probe.ok:
            detail = _join_details(
                cuda_status.detail,
                describe_msvc_environment(target_env),
                probe.detail,
                ninja_status.detail,
            )
            return CompileToolchainStatus(True, detail, cuda_status.changed)
        logger.warning("Current MSVC compiler rejected by CUDA: {}", probe.detail)

    global _CACHED_ENV_DELTA, _CACHED_STATUS
    if _CACHED_ENV_DELTA is not None and _CACHED_STATUS is not None:
        target_env.update(_CACHED_ENV_DELTA)
        return CompileToolchainStatus(
            _CACHED_STATUS.ok,
            _CACHED_STATUS.detail,
            changed=bool(_CACHED_ENV_DELTA),
        )

    before = dict(target_env)
    status = load_msvc_environment(target_env)
    if status.ok:
        cuda_status = discover_cuda_environment(target_env)
        _CACHED_ENV_DELTA = {
            key: value
            for key, value in target_env.items()
            if before.get(key) != value
        }
        detail = _join_details(cuda_status.detail, status.detail, ninja_status.detail)
        _CACHED_STATUS = CompileToolchainStatus(True, detail, bool(_CACHED_ENV_DELTA))
    else:
        _CACHED_ENV_DELTA = {}
        detail = _join_details(cuda_status.detail, status.detail, ninja_status.detail)
        _CACHED_STATUS = CompileToolchainStatus(False, detail, False)
    return _CACHED_STATUS


def prepare_compile_subprocess_env(
    env: MutableMapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
    compile_requested: bool = True,
) -> dict[str, str]:
    """Return a subprocess environment prepared only when compile is requested."""

    child_env = dict(os.environ if env is None else env)
    if compile_requested:
        status = ensure_compile_environment(child_env, project_root=project_root)
        if not status.ok:
            logger.warning("torch.compile toolchain preparation: {}", status.detail)
    return child_env


def compile_environment_report(
    *,
    project_root: str | Path | None = None,
) -> CompileToolchainStatus:
    """Return the current process toolchain status for diagnostics."""

    return ensure_compile_environment(os.environ, project_root=project_root)


def _ensure_compile_cache_dirs(
    env: MutableMapping[str, str],
    project_root: str | Path | None,
) -> None:
    """Set stable local cache folders used by Inductor and Triton."""

    root_text = str(project_root or env.get("ACESTEP_PROJECT_ROOT") or "").strip()
    root = Path(root_text).expanduser().resolve() if root_text else Path.cwd()
    cache_root = root / ".cache" / "acestep" / "torch_compile"
    inductor = cache_root / "inductor"
    triton = cache_root / "triton"
    inductor.mkdir(parents=True, exist_ok=True)
    triton.mkdir(parents=True, exist_ok=True)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor))
    env.setdefault("TRITON_CACHE_DIR", str(triton))


def _join_details(*details: str) -> str:
    """Join diagnostic detail fragments without empty entries."""

    return "; ".join(detail for detail in details if detail)
