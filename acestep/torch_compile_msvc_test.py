"""Tests for Visual Studio toolchain selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.torch_compile_compiler_probe import CompilerProbeStatus
from acestep.torch_compile_msvc import (
    _scripts_for_vs_root,
    describe_msvc_environment,
    load_msvc_environment,
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
        self.assertTrue(any(args == "-arch=amd64 -host_arch=amd64" for _, args in scripts))

    def test_describe_msvc_environment_reports_toolset_from_cl_path(self) -> None:
        """Existing PATH compiler diagnostics should include the MSVC version."""

        cl_path = r"C:\VS\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe"
        with patch("acestep.torch_compile_msvc.shutil.which", return_value=cl_path):
            detail = describe_msvc_environment({"PATH": "unused"})

        self.assertIn("14.40.33807", detail)
        self.assertIn("cl.exe", detail)

    def test_load_msvc_environment_skips_cuda_rejected_toolset(self) -> None:
        """MSVC loading should continue when nvcc rejects a candidate toolset."""

        first_script = Path(r"C:\VS\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
        second_script = Path(r"C:\VS\2022\Community\Common7\Tools\VsDevCmd.bat")

        def _fake_script_env(_script: Path, args: str) -> dict[str, str]:
            version = args.rsplit("=", 1)[-1]
            return {"PATH": rf"C:\VS\VC\Tools\MSVC\{version}\bin", "VCToolsVersion": version}

        env = {"PATH": ""}
        with patch(
            "acestep.torch_compile_msvc._candidate_msvc_scripts",
            return_value=[
                (first_script, "-vcvars_ver=14.44.35207"),
                (second_script, "-vcvars_ver=14.40.33807"),
            ],
        ), patch(
            "acestep.torch_compile_msvc._environment_from_vc_script",
            side_effect=_fake_script_env,
        ), patch(
            "acestep.torch_compile_msvc.has_cl_exe",
            return_value=True,
        ), patch(
            "acestep.torch_compile_msvc.probe_cuda_compiler",
            side_effect=[
                CompilerProbeStatus(False, "unsupported Visual Studio version"),
                CompilerProbeStatus(True, "CUDA host compiler probe passed"),
            ],
        ):
            status = load_msvc_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual("14.40.33807", env["VCToolsVersion"])
        self.assertIn("CUDA host compiler probe passed", status.detail)


if __name__ == "__main__":
    unittest.main()
