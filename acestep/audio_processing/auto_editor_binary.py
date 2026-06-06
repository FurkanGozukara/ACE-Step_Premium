"""Auto-Editor binary resolution with mirrored download fallback."""

from __future__ import annotations

import platform
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger


GITHUB_AUTO_EDITOR_URL = (
    "https://github.com/WyattBlue/auto-editor/releases/download/{version}/{asset}"
)
HUGGINGFACE_AUTO_EDITOR_URL = (
    "https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/{asset}"
)


def ensure_auto_editor_binary() -> Path | None:
    """Return a local Auto-Editor executable, downloading it with fallback if needed."""

    package_dir, version = _auto_editor_package_info()
    if package_dir is None or not version:
        return None

    asset, local_name = _asset_name()
    binary_path = package_dir / "bin" / local_name
    if binary_path.exists():
        if _binary_version(binary_path) == version:
            return binary_path
        binary_path.unlink()

    return _download_binary(binary_path, asset, version)


def _auto_editor_package_info() -> tuple[Path | None, str | None]:
    """Return the installed auto_editor package directory and version."""

    try:
        import auto_editor
    except ImportError:
        return None, None

    module_file = getattr(auto_editor, "__file__", None)
    version = str(getattr(auto_editor, "__version__", "") or "")
    if not module_file:
        return None, version
    return Path(module_file).resolve().parent, version


def _asset_name(
    system: str | None = None,
    machine: str | None = None,
) -> tuple[str, str]:
    """Return the upstream asset name and local executable name for this platform."""

    system_name = (system or platform.system()).lower()
    machine_name = (machine or platform.machine()).lower()
    is_arm64 = machine_name in {"arm64", "aarch64"}

    if system_name == "windows":
        asset = "auto-editor-windows-aarch64.exe" if is_arm64 else "auto-editor-windows-x86_64.exe"
        return asset, "auto-editor.exe"
    if system_name == "darwin":
        asset = "auto-editor-macos-arm64" if is_arm64 else "auto-editor-macos-x86_64"
        return asset, "auto-editor"
    if system_name == "linux":
        if machine_name in {"armv7l", "armv7"}:
            return "auto-editor-linux-armv7", "auto-editor"
        asset = "auto-editor-linux-aarch64" if is_arm64 else "auto-editor-linux-x86_64"
        return asset, "auto-editor"

    raise RuntimeError(f"Unsupported platform: {system_name} {machine_name}")


def _binary_version(binary_path: Path) -> str | None:
    """Return the executable version reported by Auto-Editor."""

    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout.strip()


def _download_binary(binary_path: Path, asset: str, version: str) -> Path:
    """Download an Auto-Editor binary from GitHub, then Hugging Face if needed."""

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = binary_path.with_name(f"{binary_path.name}.download")
    failures: list[str] = []

    for url in _download_urls(asset, version):
        try:
            _unlink_if_exists(temp_path)
            logger.info("Downloading Auto-Editor {} from {}", version, url)
            urllib.request.urlretrieve(url, str(temp_path))
            temp_path.replace(binary_path)
            binary_path.chmod(0o755)
            return binary_path
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"{url}: {exc}")
            _unlink_if_exists(temp_path)

    detail = "; ".join(failures) if failures else "no download URLs were available"
    raise RuntimeError(f"auto-editor binary download failed: {detail}")


def _download_urls(asset: str, version: str) -> tuple[str, str]:
    """Return primary and fallback URLs for an Auto-Editor release asset."""

    return (
        GITHUB_AUTO_EDITOR_URL.format(version=version, asset=asset),
        HUGGINGFACE_AUTO_EDITOR_URL.format(asset=asset),
    )


def _unlink_if_exists(path: Path) -> None:
    """Remove a temporary download file if it exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass
