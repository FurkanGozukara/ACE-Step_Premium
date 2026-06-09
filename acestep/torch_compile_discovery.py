"""Cross-platform toolchain discovery for optional ``torch.compile``."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, MutableMapping

from acestep.torch_compile_compiler_probe import probe_cuda_compiler
from acestep.torch_compile_cuda_toolkit import select_cuda_toolkit_root


@dataclass(frozen=True)
class ExecutableStatus:
    """Describe one discovered compiler or toolkit executable."""

    ok: bool
    detail: str
    path: str = ""


@dataclass(frozen=True)
class CudaDiscoveryStatus:
    """Describe CUDA toolkit discovery and environment changes."""

    ok: bool
    detail: str
    root: str = ""
    changed: bool = False


def discover_cuda_environment(env: MutableMapping[str, str]) -> CudaDiscoveryStatus:
    """Discover CUDA Toolkit folders, update ``env``, and return status."""

    before_path = env.get("PATH", "")
    before_cuda_home = env.get("CUDA_HOME")
    before_cuda_path = env.get("CUDA_PATH")
    root = select_cuda_toolkit_root(env, sys.platform)
    if root is None:
        return CudaDiscoveryStatus(False, "CUDA Toolkit not found")

    env["CUDA_HOME"] = str(root)
    env["CUDA_PATH"] = str(root)
    _prepend_existing_paths(env, _cuda_path_additions(root))
    changed = (
        env.get("PATH", "") != before_path
        or env.get("CUDA_HOME") != before_cuda_home
        or env.get("CUDA_PATH") != before_cuda_path
    )
    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc_path:
        return CudaDiscoveryStatus(True, f"CUDA Toolkit found at {root}", str(root), changed)
    return CudaDiscoveryStatus(
        True,
        f"CUDA root found at {root}, nvcc not present",
        str(root),
        changed,
    )


def discover_posix_compiler(env: MutableMapping[str, str]) -> ExecutableStatus:
    """Find a Linux/macOS C++ compiler, update ``CC``/``CXX``, and return status."""

    if sys.platform == "win32":
        return ExecutableStatus(True, "Windows compiler discovery handled by MSVC")

    existing_cxx = env.get("CXX")
    if existing_cxx and _which_or_file(existing_cxx, env):
        probe = probe_cuda_compiler(env, platform_name=sys.platform, compiler_path=existing_cxx)
        if probe.ok:
            return ExecutableStatus(
                True,
                f"CXX already set to {existing_cxx}; {probe.detail}",
                existing_cxx,
            )

    _prepend_existing_paths(env, _posix_compiler_path_candidates(env))
    failures: list[str] = []
    for cxx, cc in _posix_compiler_candidates():
        cxx_path = shutil.which(cxx, path=env.get("PATH"))
        if not cxx_path:
            continue
        cc_path = shutil.which(cc, path=env.get("PATH")) if cc else ""
        candidate_env = dict(env)
        candidate_env["CXX"] = cxx_path
        if cc_path:
            candidate_env["CC"] = cc_path
        probe = probe_cuda_compiler(
            candidate_env,
            platform_name=sys.platform,
            compiler_path=cxx_path,
        )
        if probe.ok:
            env["CXX"] = cxx_path
            if cc_path:
                env["CC"] = cc_path
            return ExecutableStatus(
                True,
                f"C++ compiler found: {cxx_path}; {probe.detail}",
                cxx_path,
            )
        failures.append(f"{cxx_path}: {probe.detail}")

    detail = "No compatible C++ compiler found. Install gcc/g++ or clang/clang++."
    if failures:
        detail += f" Last probe: {failures[-1]}"
    return ExecutableStatus(False, detail)


def discover_ninja(env: MutableMapping[str, str]) -> ExecutableStatus:
    """Find ``ninja`` when it is available for extension builds."""

    ninja = shutil.which("ninja", path=env.get("PATH"))
    if ninja:
        return ExecutableStatus(True, f"ninja found: {ninja}", ninja)
    return ExecutableStatus(False, "ninja not found")


def _cuda_path_additions(root: Path) -> list[Path]:
    additions = [root / "bin"]
    if sys.platform == "win32":
        additions.append(root / "libnvvp")
    return additions


def _posix_compiler_path_candidates(env: MutableMapping[str, str]) -> list[Path]:
    candidates = [Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin")]
    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.insert(0, Path(conda_prefix) / "bin")
    return candidates


def _posix_compiler_candidates() -> Iterable[tuple[str, str]]:
    versioned_gcc = [(f"g++-{version}", f"gcc-{version}") for version in range(15, 7, -1)]
    versioned_clang = [
        (f"clang++-{version}", f"clang-{version}") for version in range(20, 9, -1)
    ]
    return (
        ("g++", "gcc"),
        *versioned_gcc,
        ("clang++", "clang"),
        *versioned_clang,
        ("c++", "cc"),
    )


def _prepend_existing_paths(env: MutableMapping[str, str], paths: Iterable[Path]) -> None:
    existing = env.get("PATH", "").split(os.pathsep)
    normalized = {str(Path(part)).casefold() for part in existing if part}
    additions: list[str] = []
    for path in paths:
        if not path.is_dir():
            continue
        text = str(path)
        key = text.casefold()
        if key in normalized:
            continue
        additions.append(text)
        normalized.add(key)
    if additions:
        env["PATH"] = os.pathsep.join(additions + existing)


def _which_or_file(executable: str, env: MutableMapping[str, str]) -> bool:
    path = Path(executable)
    if path.is_file():
        return True
    return shutil.which(executable, path=env.get("PATH")) is not None
