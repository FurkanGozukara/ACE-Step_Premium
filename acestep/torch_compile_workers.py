"""Worker-pool settings for optional TorchInductor compilation."""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, MutableMapping


MIN_COMPILE_THREADS = 1
MAX_COMPILE_THREADS = 32
DEFAULT_COMPILE_THREADS = 8
COMPILE_THREADS_ENV = "TORCHINDUCTOR_COMPILE_THREADS"
WORKER_START_ENV = "TORCHINDUCTOR_WORKER_START"
_SUPPORTED_WORKER_START_METHODS = {"subprocess", "fork", "spawn"}

_CONFIG_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompileWorkerSettings:
    """Resolved TorchInductor worker settings for one LM runtime."""

    threads: int
    worker_start: str


@dataclass(frozen=True)
class CompileWorkerWarmupHandle:
    """In-flight worker-pool startup that can overlap eager LM prefill work."""

    threads: int
    worker_start: str
    started_at: float
    pool: Any = None
    futures: tuple[Any, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class CompileWorkerWarmupResult:
    """Observed worker-pool readiness before the first compiled decode."""

    requested_threads: int
    active_workers: int
    worker_start: str
    elapsed_seconds: float
    ready: bool
    detail: str


def normalize_compile_threads(
    value: Any = None,
    *,
    env: MutableMapping[str, str] | None = None,
) -> int:
    """Clamp a UI/environment thread count to the supported range."""

    if value is None or str(value).strip() == "":
        source_env = os.environ if env is None else env
        value = source_env.get(COMPILE_THREADS_ENV, DEFAULT_COMPILE_THREADS)
    try:
        threads = int(round(float(value)))
    except (TypeError, ValueError):
        threads = DEFAULT_COMPILE_THREADS
    return max(MIN_COMPILE_THREADS, min(MAX_COMPILE_THREADS, threads))


def prepare_compile_worker_env(
    env: MutableMapping[str, str],
    value: Any = None,
    *,
    platform: str | None = None,
) -> CompileWorkerSettings:
    """Write portable Inductor worker settings into an environment mapping."""

    resolved_platform = sys.platform if platform is None else platform
    threads = normalize_compile_threads(value, env=env)
    worker_start = "spawn" if resolved_platform == "win32" else str(
        env.get(WORKER_START_ENV, "subprocess")
    ).strip() or "subprocess"
    if worker_start not in _SUPPORTED_WORKER_START_METHODS:
        worker_start = "subprocess"
    env[COMPILE_THREADS_ENV] = str(threads)
    env[WORKER_START_ENV] = worker_start
    return CompileWorkerSettings(threads=threads, worker_start=worker_start)


def configure_compile_workers(value: Any = None) -> CompileWorkerSettings:
    """Configure Inductor workers before the first compiled LM invocation.

    PyTorch defaults to one compile thread on Windows. Its ``subprocess`` pool
    also relies on ``pass_fds``, which Windows does not support, so parallel
    compilation must use the standard multiprocessing ``spawn`` pool there.
    """

    with _CONFIG_LOCK:
        settings = prepare_compile_worker_env(os.environ, value)
        _apply_loaded_inductor_config(settings.threads, settings.worker_start)

    return settings


def _compile_worker_ready_probe(delay_seconds: float) -> int:
    """Keep a worker busy briefly so the executor starts its full pool."""

    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return os.getpid()


def _load_async_compile_runtime() -> tuple[Any, int]:
    """Load Inductor's pool lazily after toolchain environment discovery."""

    from torch._inductor.async_compile import AsyncCompile, get_compile_threads

    return AsyncCompile, int(get_compile_threads())


def start_compile_worker_warmup(value: Any = None) -> CompileWorkerWarmupHandle:
    """Start every configured Inductor worker without blocking the caller."""

    settings = configure_compile_workers(value)
    threads = settings.threads
    worker_start = settings.worker_start
    started_at = time.perf_counter()
    if threads <= 1:
        return CompileWorkerWarmupHandle(threads, worker_start, started_at)

    try:
        async_compile, configured_threads = _load_async_compile_runtime()
        if configured_threads != threads:
            return CompileWorkerWarmupHandle(
                threads,
                worker_start,
                started_at,
                error=(
                    "TorchInductor worker mismatch after configuration: "
                    f"requested {threads}, runtime reports {configured_threads}"
                ),
            )
        previous_warnings = os.environ.get("PYTHONWARNINGS")
        worker_warning_filters = (
            "ignore:The pynvml package is deprecated:FutureWarning",
            "ignore:<enum 'KernelPreference'> is an Enum subclass:FutureWarning",
            "ignore:<enum 'ScaleCalculationMode'> is an Enum subclass:FutureWarning",
        )
        os.environ["PYTHONWARNINGS"] = ",".join(
            ([previous_warnings] if previous_warnings else [])
            + list(worker_warning_filters)
        )
        try:
            pool = async_compile.process_pool()
            futures = tuple(
                pool.submit(_compile_worker_ready_probe, 0.2)
                for _ in range(threads)
            )
            # Prime Inductor's own readiness future while eager Phase-1 and
            # Dynamo graph capture are still able to overlap process startup.
            async_compile.use_process_pool()
        finally:
            if previous_warnings is None:
                os.environ.pop("PYTHONWARNINGS", None)
            else:
                os.environ["PYTHONWARNINGS"] = previous_warnings
        return CompileWorkerWarmupHandle(
            threads,
            worker_start,
            started_at,
            pool=pool,
            futures=futures,
        )
    except Exception as exc:
        return CompileWorkerWarmupHandle(
            threads,
            worker_start,
            started_at,
            error=f"{type(exc).__name__}: {exc}",
        )


def finish_compile_worker_warmup(
    handle: CompileWorkerWarmupHandle,
    *,
    timeout_seconds: float = 120.0,
) -> CompileWorkerWarmupResult:
    """Wait for startup probes and report how many workers became active."""

    if handle.error:
        return _warmup_result(handle, 0, False, handle.error)
    if handle.threads <= 1:
        return _warmup_result(handle, handle.threads, True, "serial compilation")

    deadline = time.perf_counter() + max(0.1, float(timeout_seconds))
    worker_pids: set[int] = set()
    try:
        for future in handle.futures:
            remaining = max(0.0, deadline - time.perf_counter())
            worker_pids.add(int(future.result(timeout=remaining)))
    except (FuturesTimeoutError, TimeoutError) as exc:
        return _warmup_result(
            handle,
            len(worker_pids),
            False,
            f"worker startup timed out: {exc}",
        )
    except Exception as exc:
        return _warmup_result(
            handle,
            len(worker_pids),
            False,
            f"worker startup failed: {type(exc).__name__}: {exc}",
        )

    active_workers = len(worker_pids)
    processes = getattr(handle.pool, "_processes", None)
    if isinstance(processes, dict):
        active_workers = max(
            active_workers,
            sum(
                1
                for process in processes.values()
                if process is not None and process.is_alive()
            ),
        )
    try:
        async_compile, _configured_threads = _load_async_compile_runtime()
        while not async_compile.use_process_pool():
            if time.perf_counter() >= deadline:
                return _warmup_result(
                    handle,
                    active_workers,
                    False,
                    "workers started but Inductor did not mark the pool ready",
                )
            time.sleep(0.01)
    except Exception as exc:
        return _warmup_result(
            handle,
            active_workers,
            False,
            f"Inductor pool readiness check failed: {type(exc).__name__}: {exc}",
        )
    return _warmup_result(
        handle,
        active_workers,
        True,
        f"worker probes completed on {active_workers} process(es)",
    )


def _warmup_result(
    handle: CompileWorkerWarmupHandle,
    active_workers: int,
    ready: bool,
    detail: str,
) -> CompileWorkerWarmupResult:
    return CompileWorkerWarmupResult(
        requested_threads=handle.threads,
        active_workers=active_workers,
        worker_start=handle.worker_start,
        elapsed_seconds=max(0.0, time.perf_counter() - handle.started_at),
        ready=ready,
        detail=detail,
    )


def _apply_loaded_inductor_config(threads: int, worker_start: str) -> None:
    """Update an already-imported Inductor config and recycle idle pools."""

    try:
        import torch._inductor.config as inductor_config
    except (ImportError, AttributeError):
        return

    current_threads = int(getattr(inductor_config, "compile_threads", threads) or threads)
    current_start = str(getattr(inductor_config, "worker_start_method", worker_start))
    if current_threads != threads or current_start != worker_start:
        try:
            from torch._inductor.async_compile import (
                AsyncCompile,
                shutdown_compile_workers,
            )
        except (ImportError, AttributeError):
            pass
        else:
            shutdown_compile_workers()
            if AsyncCompile.pool.cache_info().currsize:
                AsyncCompile.pool().shutdown(wait=True)
                AsyncCompile.pool.cache_clear()

    # Assign explicitly even when the environment-derived values appear equal.
    # Config modules can have a stale Windows default cached while their env vars
    # already show the requested values.
    inductor_config.compile_threads = threads
    inductor_config.worker_start_method = worker_start


__all__ = [
    "CompileWorkerSettings",
    "CompileWorkerWarmupHandle",
    "CompileWorkerWarmupResult",
    "DEFAULT_COMPILE_THREADS",
    "MAX_COMPILE_THREADS",
    "MIN_COMPILE_THREADS",
    "configure_compile_workers",
    "finish_compile_worker_warmup",
    "normalize_compile_threads",
    "prepare_compile_worker_env",
    "start_compile_worker_warmup",
]
