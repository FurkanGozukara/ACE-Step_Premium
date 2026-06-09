"""Tests for optional torch.compile runtime wrappers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from acestep.torch_compile_runtime import compile_module_callable, compile_module_forward
from acestep.torch_compile_toolchain import CompileToolchainStatus


class TorchCompileRuntimeTests(unittest.TestCase):
    """Verify compile requests stay optional and recover from failures."""

    def test_disabled_compile_request_is_logged(self) -> None:
        """Disabled compile options should still emit a console status."""

        module = torch.nn.Linear(2, 2)

        with patch("acestep.torch_compile_callable.log_compile_status") as log_status:
            result = compile_module_forward(
                module,
                label="disabled-test",
                enabled=False,
            )

        self.assertFalse(result.requested)
        self.assertFalse(result.compiled)
        self.assertEqual("disabled by user option", result.detail)
        self.assertEqual("disabled", log_status.call_args.kwargs["status"])

    def test_non_cuda_module_is_not_compiled(self) -> None:
        """CPU modules should be left in eager mode."""

        module = torch.nn.Linear(2, 2)

        result = compile_module_forward(module, label="cpu-test")

        self.assertFalse(result.compiled)
        self.assertEqual("device is cpu", result.detail)
        self.assertFalse(module._acestep_torch_compiled)

    def test_first_successful_forward_is_logged_as_verified(self) -> None:
        """A lazy compiled forward should report when it first succeeds."""

        module = torch.nn.Linear(2, 1)

        def fake_compile(fn, **_kwargs):
            """Return a compiled callable that delegates to the original."""

            def _compiled(*args, **kwargs):
                return fn(*args, **kwargs)

            return _compiled

        with patch(
            "acestep.torch_compile_callable.module_device_type",
            return_value="cuda",
        ), patch(
            "acestep.torch_compile_callable.ensure_compile_environment",
            return_value=CompileToolchainStatus(True, "ready"),
        ), patch("torch.compile", side_effect=fake_compile), patch(
            "acestep.torch_compile_callable.log_compile_status"
        ) as log_status:
            result = compile_module_forward(module, label="success-test")
            output = module(torch.ones(1, 2))

        self.assertTrue(result.compiled)
        self.assertEqual((1, 1), tuple(output.shape))
        self.assertTrue(module._acestep_torch_compile_verified)
        statuses = [call.kwargs["status"] for call in log_status.call_args_list]
        self.assertIn("setup_ready", statuses)
        self.assertIn("first_forward_ok", statuses)

    def test_non_forward_callable_can_be_compiled(self) -> None:
        """Callable attributes such as VAE decode should be compiled and verified."""

        class DecodeModule(torch.nn.Module):
            """Small module with a decode callable for compile tests."""

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)

            def decode(self, tensor):
                """Return a simple object with a sample tensor."""

                return type("DecodeOutput", (), {"sample": self.linear(tensor)})()

        module = DecodeModule()

        def fake_compile(fn, **_kwargs):
            """Return a compiled callable that delegates to the original."""

            def _compiled(*args, **kwargs):
                return fn(*args, **kwargs)

            return _compiled

        with patch(
            "acestep.torch_compile_callable.module_device_type",
            return_value="cuda",
        ), patch(
            "acestep.torch_compile_callable.ensure_compile_environment",
            return_value=CompileToolchainStatus(True, "ready"),
        ), patch("torch.compile", side_effect=fake_compile), patch(
            "acestep.torch_compile_callable.log_compile_status"
        ) as log_status:
            result = compile_module_callable(
                module,
                attribute_name="decode",
                label="decode-test",
            )
            output = module.decode(torch.ones(1, 2))

        self.assertTrue(result.compiled)
        self.assertEqual((1, 1), tuple(output.sample.shape))
        self.assertTrue(module._acestep_torch_compile_verified)
        statuses = [call.kwargs["status"] for call in log_status.call_args_list]
        self.assertIn("setup_ready", statuses)
        self.assertIn("first_decode_ok", statuses)

    def test_first_forward_failure_restores_eager_forward(self) -> None:
        """A compiled-forward failure should fall back to the original forward."""

        module = torch.nn.Linear(2, 1)
        original_forward = module.forward

        def fake_compile(_fn, **_kwargs):
            """Return a compiled callable that fails on first execution."""

            def _compiled(*_args, **_inner_kwargs):
                raise RuntimeError("compile graph failed")

            return _compiled

        with patch(
            "acestep.torch_compile_callable.module_device_type",
            return_value="cuda",
        ), patch(
            "acestep.torch_compile_callable.ensure_compile_environment",
            return_value=CompileToolchainStatus(True, "ready"),
        ), patch("torch.compile", side_effect=fake_compile):
            result = compile_module_forward(module, label="fallback-test")

        self.assertTrue(result.compiled)
        output = module(torch.ones(1, 2))

        self.assertEqual((1, 1), tuple(output.shape))
        self.assertIs(module.forward.__func__, original_forward.__func__)
        self.assertFalse(module._acestep_torch_compiled)
        self.assertIn("first compiled forward failed", module._acestep_torch_compile_detail)


if __name__ == "__main__":
    unittest.main()
