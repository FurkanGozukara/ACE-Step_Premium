"""Tests for the reusable high-level torch.compile integration helpers."""

from __future__ import annotations

import unittest
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch_compile_toolchain.environment import CompileToolchainStatus
from torch_compile_toolchain.runtime import compile_callable, compile_module_callable


class PortableTorchCompileRuntimeTests(unittest.TestCase):
    def test_disabled_compile_returns_original_callable(self) -> None:
        def target(value):
            return value + 1

        result = compile_callable(target, enabled=False)

        self.assertFalse(result.requested)
        self.assertFalse(result.compiled)
        self.assertIs(target, result.callable)

    def test_first_compiled_call_marks_result_verified(self) -> None:
        def target(value):
            return value + 1

        def _compile(callable_target, **_kwargs):
            return lambda value: callable_target(value)

        with patch(
            "torch_compile_toolchain.runtime.ensure_compile_environment",
            return_value=CompileToolchainStatus(True, "ready"),
        ), patch("torch.compile", side_effect=_compile):
            result = compile_callable(target)
            output = result.callable(2)

        self.assertEqual(3, output)
        self.assertTrue(result.compiled)
        self.assertTrue(result.verified)
        self.assertIn("first compiled call succeeded", result.detail)

    def test_failed_cold_compile_call_falls_back_to_eager(self) -> None:
        def target(value):
            return value + 1

        def _compile(_target, **_kwargs):
            def _failed(_value):
                raise RuntimeError("compiler failure")

            return _failed

        with patch(
            "torch_compile_toolchain.runtime.ensure_compile_environment",
            return_value=CompileToolchainStatus(True, "ready"),
        ), patch("torch.compile", side_effect=_compile):
            result = compile_callable(target)
            first = result.callable(2)
            second = result.callable(4)

        self.assertEqual(3, first)
        self.assertEqual(5, second)
        self.assertFalse(result.compiled)
        self.assertFalse(result.verified)
        self.assertIn("eager fallback", result.detail)

    def test_module_callable_is_replaced_in_place(self) -> None:
        module = SimpleNamespace(forward=lambda value: value * 2)

        with patch(
            "torch_compile_toolchain.runtime.compile_callable",
            return_value=SimpleNamespace(callable=lambda value: value * 3),
        ):
            result = compile_module_callable(module)

        self.assertEqual(9, module.forward(3))
        self.assertIs(result.callable, module.forward)

    def test_package_can_be_copied_and_imported_without_acestep(self) -> None:
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "torch_compile_toolchain"
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*_test.py"),
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = temp_dir
            completed = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-c",
                    "import torch_compile_toolchain as t; print(t.__version__)",
                ],
                cwd=temp_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual("1.0.0", completed.stdout.strip())


if __name__ == "__main__":
    unittest.main()
