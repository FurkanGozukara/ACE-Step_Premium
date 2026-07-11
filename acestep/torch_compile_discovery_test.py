"""Tests for torch.compile toolchain discovery helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch_compile_toolchain.compiler_probe import CompilerProbeStatus
from torch_compile_toolchain.cuda_toolkit import _looks_like_cuda_root
from torch_compile_toolchain.discovery import (
    discover_cuda_environment,
    discover_ninja,
    discover_posix_compiler,
)
from torch_compile_toolchain.environment import (
    CompileToolchainStatus,
    ensure_compile_environment,
)
from torch_compile_toolchain.msvc import MsvcLoadStatus


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

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                return_value="",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cuda_root.resolve()), env["CUDA_PATH"])
        self.assertIn(str(cuda_bin), env["PATH"])

    def test_generic_lib64_directory_is_not_a_cuda_toolkit(self) -> None:
        """Linux /usr-style roots need CUDA artifacts, not merely a lib64 folder."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "lib64").mkdir()

            detected = _looks_like_cuda_root(root, "linux")

        self.assertFalse(detected)

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

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                return_value="13.0",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(expected_root, env["CUDA_HOME"])
        self.assertEqual(expected_root, env["CUDA_PATH"])
        self.assertIn(str(cuda_13_bin), env["PATH"])

    def test_selected_cuda_bin_is_prioritized_over_stale_path_entry(self) -> None:
        """PATH must execute nvcc from the toolkit selected for the Torch build."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_12 = Path(temp_dir) / "v12.8"
            cuda_13 = Path(temp_dir) / "v13.0"
            cuda_12_bin = _make_cuda_root(cuda_12)
            cuda_13_bin = _make_cuda_root(cuda_13)
            env = {
                "CUDA_HOME": str(cuda_13),
                "PATH": str(cuda_12_bin),
            }

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                return_value="13.0",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cuda_13_bin), env["PATH"].split(os.pathsep)[0])

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

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                return_value="13",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(expected_root, env["CUDA_HOME"])
        self.assertEqual(expected_root, env["CUDA_PATH"])

    def test_cuda_discovery_uses_nearest_minor_when_exact_is_unavailable(self) -> None:
        """A same-major toolkit closest to Torch is safer than an arbitrary newest one."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_13_2 = Path(temp_dir) / "v13.2"
            cuda_13_1 = Path(temp_dir) / "v13.1"
            _make_cuda_root(cuda_13_2)
            _make_cuda_root(cuda_13_1)
            expected_root = str(cuda_13_1.resolve())
            env = {
                "CUDA_HOME": str(cuda_13_2),
                "CUDA_PATH": str(cuda_13_1),
                "PATH": "",
            }

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                return_value="13.0",
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(expected_root, env["CUDA_HOME"])
        self.assertEqual(expected_root, env["CUDA_PATH"])

    def test_cuda_discovery_accepts_versioned_installer_variable(self) -> None:
        """NVIDIA's CUDA_PATH_Vxx_x variables should be discovered automatically."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_root = Path(temp_dir) / "v13.1"
            _make_cuda_root(cuda_root)
            expected_root = str(cuda_root.resolve())
            env = {"CUDA_PATH_V13_1": str(cuda_root), "PATH": ""}

            with patch("torch_compile_toolchain.discovery.sys.platform", "win32"), patch(
                "torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                return_value="13.0",
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

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"):
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

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.discovery.probe_cuda_compiler",
                side_effect=[
                    CompilerProbeStatus(False, "unsupported host compiler"),
                    CompilerProbeStatus(True, "CUDA host compiler probe passed"),
                ],
            ):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertIn("g++-15", env["CXX"])
        self.assertIn("CUDA host compiler probe passed", status.detail)

    def test_posix_compiler_honors_cuda_host_cxx(self) -> None:
        """CUDAHOSTCXX should work even when CXX is not already configured."""

        with tempfile.TemporaryDirectory() as temp_dir:
            compiler_bin = Path(temp_dir) / "bin"
            compiler_bin.mkdir(parents=True)
            cxx = compiler_bin / "g++"
            cc = compiler_bin / "gcc"
            _touch_executable(cxx)
            _touch_executable(cc)
            env = {"CUDAHOSTCXX": str(cxx), "PATH": str(compiler_bin)}

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.discovery.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA host compiler probe passed"),
            ):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cxx.resolve()), env["CXX"])
        self.assertEqual(str(cc.resolve()), env["CC"])
        self.assertEqual(str(cxx.resolve()), env["NVCC_CCBIN"])

    def test_posix_compiler_prefers_nvcc_ccbin_override(self) -> None:
        """NVIDIA's compatibility-package override must outrank a generic CXX."""

        with tempfile.TemporaryDirectory() as temp_dir:
            compiler_bin = Path(temp_dir)
            compatible = compiler_bin / "g++-13"
            compatible_cc = compiler_bin / "gcc-13"
            fallback = compiler_bin / "g++-15"
            for path in (compatible, compatible_cc, fallback):
                _touch_executable(path)
            env = {
                "NVCC_CCBIN": str(compatible),
                "CXX": str(fallback),
                "PATH": str(compiler_bin),
            }

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.discovery.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA host compiler probe passed"),
            ):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(compatible.resolve()), env["CXX"])
        self.assertEqual(str(compatible.resolve()), env["NVCC_CCBIN"])

    def test_posix_compiler_discovers_conda_prefixed_toolchain(self) -> None:
        """Conda compiler packages use target-prefixed executable names."""

        with tempfile.TemporaryDirectory() as temp_dir:
            compiler_bin = Path(temp_dir)
            cxx_name = "x86_64-conda-linux-gnu-c++"
            cc_name = "x86_64-conda-linux-gnu-cc"
            for name in (cxx_name, f"{cxx_name}.exe", cc_name, f"{cc_name}.exe"):
                _touch_executable(compiler_bin / name)
            env = {"CONDA_PREFIX": temp_dir, "PATH": str(compiler_bin)}

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.discovery.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA host compiler probe passed"),
            ):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertIn(cxx_name, env["CXX"])
        self.assertIn(cc_name, env["CC"])

    def test_posix_compiler_discovers_future_versioned_gcc(self) -> None:
        """PATH scanning should not depend on a hard-coded maximum GCC version."""

        with tempfile.TemporaryDirectory() as temp_dir:
            compiler_bin = Path(temp_dir)
            for name in ("g++-42", "g++-42.exe", "gcc-42", "gcc-42.exe"):
                _touch_executable(compiler_bin / name)
            env = {"PATH": str(compiler_bin)}

            with patch("torch_compile_toolchain.discovery.sys.platform", "linux"), patch(
                "torch_compile_toolchain.discovery.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA host compiler probe passed"),
            ):
                status = discover_posix_compiler(env)

        self.assertTrue(status.ok)
        self.assertIn("g++-42", env["CXX"])

    def test_ninja_is_discovered_from_virtual_environment(self) -> None:
        """Ninja need not already be present on the inherited PATH."""

        with tempfile.TemporaryDirectory() as temp_dir:
            scripts = Path(temp_dir) / "Scripts"
            scripts.mkdir()
            ninja = scripts / "ninja.exe"
            _touch_executable(ninja)
            env = {"VIRTUAL_ENV": temp_dir, "PATH": ""}

            with patch("torch_compile_toolchain.discovery.sys.platform", "win32"):
                status = discover_ninja(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(ninja.resolve()), env["NINJA"])
        self.assertIn(str(scripts), env["PATH"])

    def test_ninja_is_found_in_custom_visual_studio_install(self) -> None:
        """Ninja discovery shares vswhere roots instead of assuming Program Files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            install = Path(temp_dir) / "CustomVS" / "BuildTools"
            ninja = (
                install
                / "Common7"
                / "IDE"
                / "CommonExtensions"
                / "Microsoft"
                / "CMake"
                / "Ninja"
                / "ninja.exe"
            )
            ninja.parent.mkdir(parents=True)
            _touch_executable(ninja)
            env = {"PATH": ""}

            with patch(
                "torch_compile_toolchain.discovery.sys.platform",
                "win32",
            ), patch(
                "torch_compile_toolchain.discovery.visual_studio_install_roots",
                return_value=[install],
            ):
                status = discover_ninja(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(ninja.resolve()), env["NINJA"])

    def test_windows_toolchain_checks_ninja_after_loading_msvc(self) -> None:
        """Visual Studio's bundled Ninja becomes visible only after VsDevCmd runs."""

        def _load_msvc(env) -> MsvcLoadStatus:
            env["PATH"] = "visual-studio-path"
            return MsvcLoadStatus(True, "MSVC loaded", True)

        def _find_ninja(env) -> CompileToolchainStatus:
            self.assertEqual("visual-studio-path", env["PATH"])
            return CompileToolchainStatus(True, "ninja found", False)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "torch_compile_toolchain.environment.sys.platform",
            "win32",
        ), patch(
            "torch_compile_toolchain.environment.discover_cuda_environment",
            return_value=CompileToolchainStatus(True, "CUDA ready", False),
        ), patch(
            "torch_compile_toolchain.environment.has_cl_exe",
            return_value=False,
        ), patch(
            "torch_compile_toolchain.environment.load_msvc_environment",
            side_effect=_load_msvc,
        ), patch(
            "torch_compile_toolchain.environment.discover_ninja",
            side_effect=_find_ninja,
        ):
            status = ensure_compile_environment({"PATH": ""}, project_root=temp_dir)

        self.assertTrue(status.ok)
        self.assertIn("ninja found", status.detail)

    def test_windows_toolchain_retries_after_an_installation_change(self) -> None:
        """A failed probe must not be cached for the lifetime of the running app."""

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "torch_compile_toolchain.environment.sys.platform",
            "win32",
        ), patch(
            "torch_compile_toolchain.environment.discover_cuda_environment",
            return_value=CompileToolchainStatus(True, "CUDA ready", False),
        ), patch(
            "torch_compile_toolchain.environment.has_cl_exe",
            return_value=False,
        ), patch(
            "torch_compile_toolchain.environment.load_msvc_environment",
            side_effect=[
                MsvcLoadStatus(False, "not installed", False),
                MsvcLoadStatus(True, "MSVC loaded", True),
            ],
        ) as load_msvc, patch(
            "torch_compile_toolchain.environment.discover_ninja",
            return_value=CompileToolchainStatus(True, "ninja found", False),
        ):
            first = ensure_compile_environment({"PATH": ""}, project_root=temp_dir)
            second = ensure_compile_environment({"PATH": ""}, project_root=temp_dir)

        self.assertFalse(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(2, load_msvc.call_count)

    def test_posix_toolchain_failure_is_reported(self) -> None:
        """Top-level setup should report missing POSIX compiler details."""

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "torch_compile_toolchain.environment.sys.platform",
            "linux",
        ), patch(
            "torch_compile_toolchain.environment.discover_cuda_environment",
            return_value=CompileToolchainStatus(True, "CUDA ready", True),
        ), patch(
            "torch_compile_toolchain.environment.discover_posix_compiler",
            return_value=CompileToolchainStatus(False, "compiler missing", False),
        ), patch(
            "torch_compile_toolchain.environment.discover_ninja",
            return_value=CompileToolchainStatus(False, "ninja missing", False),
        ):
            status = ensure_compile_environment({"PATH": os.environ.get("PATH", "")}, project_root=temp_dir)

        self.assertFalse(status.ok)
        self.assertIn("CUDA ready", status.detail)
        self.assertIn("compiler missing", status.detail)
        self.assertIn("ninja missing", status.detail)


if __name__ == "__main__":
    unittest.main()
