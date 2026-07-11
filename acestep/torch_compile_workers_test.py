"""Tests for portable TorchInductor worker configuration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from acestep.torch_compile_workers import (
    COMPILE_THREADS_ENV,
    DEFAULT_COMPILE_THREADS,
    WORKER_START_ENV,
    configure_compile_workers,
    finish_compile_worker_warmup,
    normalize_compile_threads,
    prepare_compile_worker_env,
    start_compile_worker_warmup,
)


class CompileWorkerSettingsTests(unittest.TestCase):
    def test_thread_count_defaults_and_clamps(self) -> None:
        self.assertEqual(
            normalize_compile_threads(None, env={}),
            DEFAULT_COMPILE_THREADS,
        )
        self.assertEqual(normalize_compile_threads(0), 1)
        self.assertEqual(normalize_compile_threads(99), 32)
        self.assertEqual(normalize_compile_threads("7.6"), 8)
        self.assertEqual(normalize_compile_threads("invalid"), DEFAULT_COMPILE_THREADS)

    def test_windows_uses_spawn_worker_pool(self) -> None:
        env = {WORKER_START_ENV: "subprocess"}

        settings = prepare_compile_worker_env(env, 12, platform="win32")

        self.assertEqual(settings.threads, 12)
        self.assertEqual(settings.worker_start, "spawn")
        self.assertEqual(env[COMPILE_THREADS_ENV], "12")
        self.assertEqual(env[WORKER_START_ENV], "spawn")

    def test_linux_preserves_supported_worker_start_override(self) -> None:
        env = {WORKER_START_ENV: "fork"}

        settings = prepare_compile_worker_env(env, 6, platform="linux")

        self.assertEqual(settings.threads, 6)
        self.assertEqual(settings.worker_start, "fork")
        self.assertEqual(env[COMPILE_THREADS_ENV], "6")
        self.assertEqual(env[WORKER_START_ENV], "fork")

    def test_linux_rejects_unknown_worker_start_override(self) -> None:
        env = {WORKER_START_ENV: "not-a-worker"}

        settings = prepare_compile_worker_env(env, 6, platform="linux")

        self.assertEqual(settings.worker_start, "subprocess")
        self.assertEqual(env[WORKER_START_ENV], "subprocess")

    def test_loaded_inductor_config_receives_resolved_settings(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "acestep.torch_compile_workers.sys.platform",
            "win32",
        ), patch(
            "acestep.torch_compile_workers._apply_loaded_inductor_config"
        ) as apply_config:
            settings = configure_compile_workers(5)

        self.assertEqual(settings.threads, 5)
        self.assertEqual(settings.worker_start, "spawn")
        apply_config.assert_called_once_with(5, "spawn")

    def test_worker_warmup_starts_and_confirms_full_pool(self) -> None:
        class _Future:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def result(self, timeout=None) -> int:
                self.assert_timeout = timeout
                return self.pid

        class _Pool:
            def __init__(self) -> None:
                self.pids = iter((101, 102, 103, 104))
                self.submit_count = 0

            def submit(self, _fn, _delay):
                self.submit_count += 1
                return _Future(next(self.pids))

        pool = _Pool()

        class _AsyncCompile:
            @staticmethod
            def process_pool():
                return pool

            @staticmethod
            def use_process_pool():
                return True

        with patch.dict(os.environ, {WORKER_START_ENV: "spawn"}), patch(
            "acestep.torch_compile_workers._load_async_compile_runtime",
            return_value=(_AsyncCompile, 4),
        ):
            handle = start_compile_worker_warmup(4)
            result = finish_compile_worker_warmup(handle)

        self.assertEqual(pool.submit_count, 4)
        self.assertTrue(result.ready)
        self.assertEqual(result.requested_threads, 4)
        self.assertEqual(result.active_workers, 4)
        self.assertEqual(result.worker_start, "spawn")


if __name__ == "__main__":
    unittest.main()
