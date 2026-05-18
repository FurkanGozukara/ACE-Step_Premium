"""Unit tests for cooperative generation cancellation state."""

import sys
import unittest
from types import SimpleNamespace

from acestep.core.generation.cancellation import (
    GenerationCancelled,
    check_generation_cancelled,
    cleanup_runtime_memory,
    generation_cancel_scope,
    is_generation_cancelled,
    request_generation_cancel,
    register_generation_subprocess,
    unregister_generation_subprocess,
)


class _FakeProcess:
    """Minimal process-like object for subprocess cancellation tests."""

    def __init__(self) -> None:
        """Initialize fake process state."""

        self.terminated = False

    def poll(self):
        """Return ``None`` while the fake process is still running."""

        return None if not self.terminated else 1

    def terminate(self) -> None:
        """Record that termination was requested."""

        self.terminated = True


class _SlowProcess(_FakeProcess):
    """Process-like object that requires force-kill after termination."""

    def __init__(self) -> None:
        """Initialize fake slow process state."""

        super().__init__()
        self.killed = False

    def poll(self):
        """Return ``None`` until the fake process is killed."""

        return 1 if self.killed else None

    def wait(self, timeout=None):
        """Simulate a worker that does not exit after terminate."""

        _ = timeout
        raise TimeoutError("still running")

    def kill(self) -> None:
        """Record that force-kill was requested."""

        self.killed = True


class GenerationCancellationTests(unittest.TestCase):
    """Validate generation cancellation state and subprocess termination."""

    def tearDown(self) -> None:
        """Clear cancellation state left by each test."""

        with generation_cancel_scope():
            pass

    def test_cancel_request_clears_when_outer_scope_finishes(self) -> None:
        """Cancellation should reset when the cancelled generation scope exits."""

        with generation_cancel_scope():
            request_generation_cancel()
            with self.assertRaises(GenerationCancelled):
                check_generation_cancelled()

        self.assertFalse(is_generation_cancelled())

        with generation_cancel_scope():
            check_generation_cancelled()
            self.assertFalse(is_generation_cancelled())

    def test_nested_scope_preserves_cancel_request_for_batch(self) -> None:
        """Nested generation scopes should not clear an outer batch cancellation."""

        with generation_cancel_scope():
            with generation_cancel_scope():
                request_generation_cancel()
            with self.assertRaises(GenerationCancelled):
                with generation_cancel_scope():
                    check_generation_cancelled()

    def test_cancel_terminates_registered_subprocess(self) -> None:
        """Cancel should terminate isolated generation workers."""

        process = _FakeProcess()
        register_generation_subprocess(process)
        try:
            request_generation_cancel()
        finally:
            unregister_generation_subprocess(process)

        self.assertTrue(process.terminated)

    def test_subprocess_only_cancel_ignores_foreground_generation(self) -> None:
        """Subprocess-only cancel should not stop in-process generation scopes."""

        with generation_cancel_scope():
            self.assertFalse(request_generation_cancel(subprocess_only=True))
            check_generation_cancelled()

        self.assertFalse(is_generation_cancelled())

    def test_subprocess_cancel_kills_worker_when_terminate_stalls(self) -> None:
        """Subprocess cancellation should force-kill workers that do not exit."""

        process = _SlowProcess()
        register_generation_subprocess(process)
        try:
            request_generation_cancel(subprocess_only=True)
        finally:
            unregister_generation_subprocess(process)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_idle_cancel_request_does_not_leave_stale_flag(self) -> None:
        """Cancel without active generation should not poison the next run."""

        self.assertFalse(request_generation_cancel())
        self.assertFalse(is_generation_cancelled())

        with generation_cancel_scope():
            check_generation_cancelled()

    def test_cleanup_runtime_memory_empties_cuda_allocator_cache(self) -> None:
        """Cleanup should release CUDA allocator caches when torch is loaded."""

        calls = []
        previous_torch = sys.modules.get("torch")
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: calls.append("empty_cache"),
                ipc_collect=lambda: calls.append("ipc_collect"),
            ),
            xpu=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: calls.append("xpu_empty_cache"),
            ),
        )
        sys.modules["torch"] = fake_torch
        try:
            cleanup_runtime_memory()
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

        self.assertIn("empty_cache", calls)
        self.assertIn("ipc_collect", calls)
        self.assertIn("xpu_empty_cache", calls)


if __name__ == "__main__":
    unittest.main()
