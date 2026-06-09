"""Tests for torch.compile toolchain discovery helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.torch_compile_compiler_probe import CompilerProbeStatus
from acestep.torch_compile_discovery import (
    discover_cuda_environment,
    discover_posix_compiler,
)
from acestep.torch_compile_toolchain import CompileToolchainStatus, ensure_compile_environment


def _touch_executable(path: Path) -> None:
    """Create a small executable-like file for PATH discovery tests."""

    path.write_text("", encoding="utf-8")


def _make_cuda_root(path: Path) -> Path:
    """Create a minimal CUDA Toolkit root for discovery tests."""

    cuda_bin = path / "bin"
    cuda_bin.mkdir(parents=True)
    _touch_executable(cuda_bin / "nvcc")
    _touch_executable(cuda_bin / "nvcc.exe")
    return cuda_bin


class TorchCompileDiscoveryTests(unittest.TestCase):
    """Verify cross-platform compiler and CUDA discovery behavior."""

    def test_cuda_home_is_added_to_environment(self) -> None:
        """CUDA_HOME roots should populate CUDA_PATH and PATH."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_root = Path(temp_dir) / "cuda"
            cuda_bin = _make_cuda_root(cuda_root)
            env = {"CUDA_HOME": str(cuda_root), "PATH": ""}

            with patch("acestep.torch_compile_discovery.sys.platform", "linux"), patch(
                "acestep.torch_compile_cuda_toolkit.torch_cuda_version",
                return_value="",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cuda_root.resolve()), env["CUDA_PATH"])
        self.assertIn(str(cuda_bin), env["PATH"])

    def test_cuda_discovery_prefers_toolkit_matching_torch_cuda(self) -> None:
        """Torch CUDA version should override a valid but mismatched CUDA_HOME."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_12 = Path(temp_dir) / "v12.8"
            cuda_13 = Path(temp_dir) / "v13.0"
            _make_cuda_root(cuda_12)
            cuda_13_bin = _make_cuda_root(cuda_13)
            expected_root = str(cuda_13.resolve())
            env = {
                "CUDA_HOME": str(cuda_12),
                "CUDA_PATH": str(cuda_13),
                "PATH": "",
            }

            with patch("acestep.torch_compile_discovery.sys.platform", "linux"), patch(
                "acestep.torch_compile_cuda_toolkit.torch_cuda_version",
                return_value="13.0",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(expected_root, env["CUDA_HOME"])
        self.assertEqual(expected_root, env["CUDA_PATH"])
        self.assertIn(str(cuda_13_bin), env["PATH"])

    def test_cuda_discovery_accepts_torch_cuda_major_version(self) -> None:
        """Torch CUDA major versions should match installed major.minor toolkits."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_12 = Path(temp_dir) / "v12.8"
            cuda_13 = Path(temp_dir) / "v13.1"
            _make_cuda_root(cuda_12)
            _make_cuda_root(cuda_13)
            expected_root = str(cuda_13.resolve())
            env = {
                "CUDA_HOME": str(cuda_12),
                "CUDA_PATH": str(cuda_13),
                "PATH": "",
            }

            with patch("acestep.torch_compile_discovery.sys.platform", "linux"), patch(
                "acestep.torch_compile_cuda_toolkit.torch_cuda_version",
                return_value="13",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(expected_root, env["CUDA_HOME"])
        self.assertEqual(expected_root, env["CUDA_PATH"])

    def test_posix_compiler_sets_cc_and_cxx_from_conda_prefix(self) -> None:
        """A compiler in CONDA_PREFIX/bin should be discovered."""

        with tempfile.TemporaryDirectory() as temp_dir:
            conda_bin = Path(temp_dir) / "bin"
            conda_bin.mkdir(parents=True)
            _touch_executable(conda_bin / "g++")
            _touch_executable(conda_bin / "g++.exe")
            _touch_executable(conda_bin / "gcc")
            _touch_executable(conda_bin / "gcc.exe")
            env = {"CONDA_PREFIX": temp_dir, "PATH": ""}

            with patch("acestep.torch_compile_discovery.sys.platform", "linux"):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertIn("g++", env["CXX"])
        self.assertIn("gcc", env["CC"])

    def test_posix_compiler_skips_cuda_rejected_candidate(self) -> None:
        """POSIX compiler selection should continue when nvcc rejects a candidate."""

        with tempfile.TemporaryDirectory() as temp_dir:
            compiler_bin = Path(temp_dir) / "bin"
            compiler_bin.mkdir(parents=True)
            for name in ("g++", "g++.exe", "gcc", "gcc.exe"):
                _touch_executable(compiler_bin / name)
            for name in ("g++-15", "g++-15.exe", "gcc-15", "gcc-15.exe"):
                _touch_executable(compiler_bin / name)
            env = {"PATH": str(compiler_bin)}

            with patch("acestep.torch_compile_discovery.sys.platform", "linux"), patch(
                "acestep.torch_compile_discovery.probe_cuda_compiler",
                side_effect=[
                    CompilerProbeStatus(False, "unsupported host compiler"),
                    CompilerProbeStatus(True, "CUDA host compiler probe passed"),
                ],
            ):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertIn("g++-15", env["CXX"])
        self.assertIn("CUDA host compiler probe passed", status.detail)

    def test_posix_toolchain_failure_is_reported(self) -> None:
        """Top-level setup should report missing POSIX compiler details."""

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "acestep.torch_compile_toolchain.sys.platform",
            "linux",
        ), patch(
            "acestep.torch_compile_toolchain.discover_cuda_environment",
            return_value=CompileToolchainStatus(True, "CUDA ready", True),
        ), patch(
            "acestep.torch_compile_toolchain.discover_posix_compiler",
            return_value=CompileToolchainStatus(False, "compiler missing", False),
        ), patch(
            "acestep.torch_compile_toolchain.discover_ninja",
            return_value=CompileToolchainStatus(False, "ninja missing", False),
        ):
            status = ensure_compile_environment({"PATH": os.environ.get("PATH", "")}, project_root=temp_dir)

        self.assertFalse(status.ok)
        self.assertIn("CUDA ready", status.detail)
        self.assertIn("compiler missing", status.detail)
        self.assertIn("ninja missing", status.detail)


if __name__ == "__main__":
    unittest.main()
