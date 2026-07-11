"""Tests for Visual Studio toolchain selection."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch_compile_toolchain.compiler_probe import CompilerProbeStatus
from torch_compile_toolchain.msvc import (
    _candidate_msvc_scripts,
    _environment_from_vc_script,
    _scripts_for_vs_root,
    _sort_msvc_candidates,
    describe_msvc_environment,
    load_msvc_environment,
    visual_studio_install_roots,
)


def _touch(path: Path) -> None:
    """Create an empty file, including parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class TorchCompileMsvcTests(unittest.TestCase):
    """Verify deterministic MSVC toolset discovery."""

    def test_newest_installed_toolset_is_first_candidate(self) -> None:
        """Candidate scripts should prefer newest installed MSVC toolsets."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch(root / "Common7" / "Tools" / "VsDevCmd.bat")
            for version in ("14.29.30133", "14.40.33807"):
                _touch(root / "VC" / "Tools" / "MSVC" / version / "bin" / "Hostx64" / "x64" / "cl.exe")

            scripts = _scripts_for_vs_root(root)

        self.assertIn("-vcvars_ver=14.40.33807", scripts[0][1])
        self.assertTrue(any("-vcvars_ver=14.40" in args for _, args in scripts))
        self.assertTrue(any(args == "-arch=amd64 -host_arch=amd64" for _, args in scripts))

    def test_candidates_are_sorted_by_toolset_across_installations(self) -> None:
        """Folder naming and edition order must not outrank the actual MSVC version."""

        older = (Path(r"C:\VS2022\VsDevCmd.bat"), "-vcvars_ver=14.44.35207")
        newer = (Path(r"C:\VS18\VsDevCmd.bat"), "-vcvars_ver=14.50.10000")

        ordered = _sort_msvc_candidates([older, newer])

        self.assertEqual(newer, ordered[0])

    def test_describe_msvc_environment_reports_toolset_from_cl_path(self) -> None:
        """Existing PATH compiler diagnostics should include the MSVC version."""

        cl_path = r"C:\VS\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe"
        with patch("torch_compile_toolchain.msvc.shutil.which", return_value=cl_path):
            detail = describe_msvc_environment({"PATH": "unused"})

        self.assertIn("14.40.33807", detail)
        self.assertIn("cl.exe", detail)

    def test_load_msvc_environment_skips_cuda_rejected_toolset(self) -> None:
        """MSVC loading should continue when nvcc rejects a candidate toolset."""

        first_script = Path(r"C:\VS\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
        second_script = Path(r"C:\VS\2022\Community\Common7\Tools\VsDevCmd.bat")

        def _fake_script_env(
            _script: Path,
            args: str,
            *,
            base_env=None,
        ) -> dict[str, str]:
            version = args.rsplit("=", 1)[-1]
            return {"PATH": rf"C:\VS\VC\Tools\MSVC\{version}\bin", "VCToolsVersion": version}

        env = {"PATH": ""}
        with patch(
            "torch_compile_toolchain.msvc._candidate_msvc_scripts",
            return_value=[
                (first_script, "-vcvars_ver=14.44.35207"),
                (second_script, "-vcvars_ver=14.40.33807"),
            ],
        ), patch(
            "torch_compile_toolchain.msvc._environment_from_vc_script",
            side_effect=_fake_script_env,
        ), patch(
            "torch_compile_toolchain.msvc.has_cl_exe",
            return_value=True,
        ), patch(
            "torch_compile_toolchain.msvc.probe_cuda_compiler",
            side_effect=[
                CompilerProbeStatus(False, "unsupported Visual Studio version"),
                CompilerProbeStatus(True, "CUDA host compiler probe passed"),
            ],
        ):
            status = load_msvc_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual("14.40.33807", env["VCToolsVersion"])
        self.assertIn("CUDA host compiler probe passed", status.detail)

    def test_vc_script_uses_cmd_compatible_string_command_line(self) -> None:
        """Quoted VS paths must not be backslash-escaped by list2cmdline."""

        script = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VsDevCmd.bat")
        completed = subprocess.CompletedProcess(
            args="",
            returncode=0,
            stdout="Path=C:\\VS\\bin\nVCToolsVersion=14.44.35207\n",
            stderr="",
        )
        with patch(
            "torch_compile_toolchain.msvc.subprocess.run",
            return_value=completed,
        ) as run:
            loaded = _environment_from_vc_script(
                script,
                "-arch=amd64 -host_arch=amd64",
                base_env={"COMSPEC": r"C:\Windows\System32\cmd.exe", "PATH": ""},
            )

        command_line = run.call_args.args[0]
        self.assertIsInstance(command_line, str)
        self.assertIn(f'call "{script}"', command_line)
        self.assertNotIn(r'\"C:\Program Files', command_line)
        self.assertEqual(r"C:\VS\bin", loaded["Path"])

    def test_mixed_case_path_from_vs_replaces_existing_path(self) -> None:
        """The ``Path`` spelling emitted by cmd must update Python's ``PATH`` key."""

        script = Path(r"C:\VS\Common7\Tools\VsDevCmd.bat")
        env = {"PATH": "original"}

        def _has_loaded_cl(candidate) -> bool:
            return candidate.get("PATH") == "developer-path" and "Path" not in candidate

        with patch(
            "torch_compile_toolchain.msvc._candidate_msvc_scripts",
            return_value=[(script, "")],
        ), patch(
            "torch_compile_toolchain.msvc._environment_from_vc_script",
            return_value={"Path": "developer-path", "VCToolsVersion": "14.44.35207"},
        ), patch(
            "torch_compile_toolchain.msvc.has_cl_exe",
            side_effect=_has_loaded_cl,
        ), patch(
            "torch_compile_toolchain.msvc.probe_cuda_compiler",
            return_value=CompilerProbeStatus(True, "CUDA host compiler probe passed"),
        ):
            status = load_msvc_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual("developer-path", env["PATH"])
        self.assertNotIn("Path", env)

    def test_vswhere_candidates_include_community_and_build_tools(self) -> None:
        """Both full Visual Studio and standalone Build Tools installations work."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            community = root / "Microsoft Visual Studio" / "2022" / "Community"
            build_tools = root / "Microsoft Visual Studio" / "2022" / "BuildTools"
            for install in (community, build_tools):
                _touch(install / "Common7" / "Tools" / "VsDevCmd.bat")
                _touch(
                    install
                    / "VC"
                    / "Tools"
                    / "MSVC"
                    / "14.44.35207"
                    / "bin"
                    / "Hostx64"
                    / "x64"
                    / "cl.exe"
                )
            with patch(
                "torch_compile_toolchain.msvc._vswhere_path",
                return_value=Path("vswhere.exe"),
            ), patch(
                "torch_compile_toolchain.msvc._vswhere_install_paths",
                return_value=[str(community), str(build_tools)],
            ), patch(
                "torch_compile_toolchain.msvc._standard_visual_studio_roots",
                return_value=[],
            ):
                candidates = _candidate_msvc_scripts({"PATH": ""})

        scripts = {str(script) for script, _ in candidates}
        self.assertTrue(any("Community" in script for script in scripts))
        self.assertTrue(any("BuildTools" in script for script in scripts))

    def test_legacy_visual_studio_vcvarsall_is_a_candidate(self) -> None:
        """Older Visual Studio layouts remain usable for matching legacy CUDA builds."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_script = root / "VC" / "vcvarsall.bat"
            _touch(legacy_script)

            scripts = _scripts_for_vs_root(root)

        self.assertIn((legacy_script, "amd64"), scripts)

    def test_programw6432_finds_64_bit_visual_studio_from_32_bit_python(self) -> None:
        """The native Program Files tree is searched even from a 32-bit process."""

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "torch_compile_toolchain.msvc._vswhere_path",
            return_value=None,
        ), patch(
            "torch_compile_toolchain.msvc._registry_visual_studio_roots",
            return_value=[],
        ):
            root = Path(temp_dir) / "Microsoft Visual Studio" / "2022" / "BuildTools"
            _touch(root / "Common7" / "Tools" / "VsDevCmd.bat")
            discovered = visual_studio_install_roots(
                {"ProgramW6432": temp_dir, "PATH": ""}
            )

        self.assertIn(root, discovered)


if __name__ == "__main__":
    unittest.main()
