"""Visual Studio C++ environment discovery for ``torch.compile`` on Windows."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

from loguru import logger

from acestep.torch_compile_compiler_probe import probe_cuda_compiler


@dataclass(frozen=True)
class MsvcLoadStatus:
    """Summarize the detected MSVC compiler environment."""

    ok: bool
    detail: str
    changed: bool = False


def has_cl_exe(env: MutableMapping[str, str]) -> bool:
    """Return whether ``cl.exe`` is resolvable from the supplied PATH."""

    return shutil.which("cl.exe", path=env.get("PATH")) is not None


def describe_msvc_environment(env: MutableMapping[str, str]) -> str:
    """Return a diagnostic string for the active MSVC compiler."""

    cl_path = shutil.which("cl.exe", path=env.get("PATH")) or "cl.exe"
    version = env.get("VCToolsVersion") or _toolset_from_cl_path(cl_path)
    if version:
        return f"MSVC compiler already on PATH ({version}, {cl_path})"
    return f"MSVC compiler already on PATH ({cl_path})"


def load_msvc_environment(env: MutableMapping[str, str]) -> MsvcLoadStatus:
    """Load the first Visual Studio compiler environment accepted by CUDA."""

    failures: list[str] = []
    for script, args in _candidate_msvc_scripts():
        loaded_env = _environment_from_vc_script(script, args)
        if not loaded_env:
            continue
        candidate_env = dict(env)
        candidate_env.update(loaded_env)
        if not has_cl_exe(candidate_env):
            continue
        probe = probe_cuda_compiler(candidate_env, platform_name="win32")
        version = loaded_env.get("VCToolsVersion")
        if probe.ok:
            env.update(loaded_env)
            detail = (
                f"MSVC {version} loaded from {script}; {probe.detail}"
                if version
                else f"MSVC loaded from {script}; {probe.detail}"
            )
            logger.info("torch.compile {}", detail)
            return MsvcLoadStatus(True, detail, True)
        label = version or str(script)
        failures.append(f"{label}: {probe.detail}")
        logger.debug("Skipping MSVC candidate for torch.compile: {}", failures[-1])
    detail = "MSVC cl.exe was not found. Install Visual Studio Build Tools with C++ workload."
    if failures:
        detail = f"No CUDA-compatible MSVC toolset found. Last probe: {failures[-1]}"
    return MsvcLoadStatus(
        False,
        detail,
        False,
    )


def _candidate_msvc_scripts() -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    vswhere = _vswhere_path()
    if vswhere is not None:
        for install_path in _vswhere_install_paths(vswhere):
            candidates.extend(_scripts_for_vs_root(Path(install_path)))

    roots: list[Path] = []
    for env_key in ("VSINSTALLDIR", "VCINSTALLDIR"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value))
    program_files = (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"))
    editions = ("BuildTools", "Community", "Professional", "Enterprise")
    for base in [Path(path) for path in program_files if path]:
        for year in ("2022", "2019"):
            for edition in editions:
                roots.append(base / "Microsoft Visual Studio" / year / edition)
    for root in roots:
        candidates.extend(_scripts_for_vs_root(root))
    return _existing_unique_scripts(candidates)


def _existing_unique_scripts(candidates: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[Path, str]] = []
    for script, args in candidates:
        key = (str(script).casefold(), args)
        if script.is_file() and key not in seen:
            unique.append((script, args))
            seen.add(key)
    return unique


def _scripts_for_vs_root(root: Path) -> list[tuple[Path, str]]:
    build_root = root / "VC" / "Auxiliary" / "Build"
    scripts = [
        (root / "Common7" / "Tools" / "VsDevCmd.bat", "-arch=amd64 -host_arch=amd64"),
        (build_root / "vcvars64.bat", ""),
        (build_root / "vcvarsall.bat", "amd64"),
        (build_root / "vcvarsall.bat", "x64"),
    ]
    candidates: list[tuple[Path, str]] = []
    for version in _msvc_toolset_versions(root):
        for script, args in scripts:
            candidates.append((script, f"{args} -vcvars_ver={version}".strip()))
    candidates.extend(scripts)
    return candidates


def _vswhere_path() -> Path | None:
    found = shutil.which("vswhere.exe")
    if found:
        return Path(found)
    path = Path(os.environ.get("ProgramFiles(x86)", "")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    return path if path.is_file() else None


def _vswhere_install_paths(vswhere: Path) -> list[str]:
    command = [
        str(vswhere),
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-sort",
        "-property",
        "installationPath",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _environment_from_vc_script(script: Path, args: str) -> dict[str, str]:
    command = f'call "{script}" {args} >nul && set'
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            ["cmd", "/d", "/s", "/c", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    return _parse_set_output(completed.stdout)


def _parse_set_output(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if not line or line.startswith("=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def _msvc_toolset_versions(root: Path) -> list[str]:
    tools_root = root / "VC" / "Tools" / "MSVC"
    if not tools_root.is_dir():
        return []
    versions = [
        path.name
        for path in tools_root.iterdir()
        if (path / "bin" / "Hostx64" / "x64" / "cl.exe").is_file()
    ]
    return sorted(versions, key=_version_key, reverse=True)


def _version_key(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.replace("-", ".").split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _toolset_from_cl_path(cl_path: str) -> str:
    parts = Path(cl_path).parts
    for index, part in enumerate(parts):
        if part.casefold() == "msvc" and index + 1 < len(parts):
            return parts[index + 1]
    return ""
