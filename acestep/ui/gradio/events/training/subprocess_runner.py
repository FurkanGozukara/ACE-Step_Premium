"""Parent-process helpers for isolated training UI workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from acestep.core.generation.subprocess_termination import terminate_generation_process

from .subprocess_control import (
    consume_training_subprocess_stop_requested,
    register_training_subprocess,
    unregister_training_subprocess,
)


_EVENT_PREFIX = "ACE_TRAINING_EVENT "
_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class TrainingSubprocessJob:
    """File paths used by one isolated training worker request."""

    work_dir: Path
    request_path: Path
    result_path: Path


def create_training_subprocess_job(project_root: str | Path) -> TrainingSubprocessJob:
    """Create an isolated job directory under the project cache."""

    work_dir = (
        Path(project_root).resolve()
        / ".cache"
        / "acestep"
        / "training_subprocess_jobs"
        / uuid.uuid4().hex
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    return TrainingSubprocessJob(
        work_dir=work_dir,
        request_path=work_dir / "request.json",
        result_path=work_dir / "result.json",
    )


def stream_training_subprocess_job(
    payload: dict[str, Any],
    job: TrainingSubprocessJob,
) -> Iterator[dict[str, Any]]:
    """Launch a worker and stream status events followed by the result event."""

    _write_json(job.request_path, payload)
    process = _start_worker(payload["project_root"], job)
    register_training_subprocess(process)
    try:
        yield from _read_worker_events(process)
    finally:
        return_code = _wait_for_process_exit(process)
        if process.stdout:
            process.stdout.close()
        unregister_training_subprocess(process)
    if consume_training_subprocess_stop_requested():
        yield {
            "kind": "result",
            "result": {
                "success": True,
                "status": "Isolated training subprocess stopped.",
                "log": "Isolated training subprocess stopped.",
            },
        }
        return

    result = _read_result(job.result_path, return_code)
    yield {"kind": "result", "result": result}


def _start_worker(project_root: str, job: TrainingSubprocessJob) -> subprocess.Popen:
    """Start the CLI worker with JSON request/result paths."""

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("ACESTEP_PROJECT_ROOT", str(project_root))
    command = [
        sys.executable,
        "-m",
        "acestep.ui.gradio.events.training.subprocess_worker",
        "--request",
        str(job.request_path),
        "--result",
        str(job.result_path),
    ]
    return subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


def _read_worker_events(process: subprocess.Popen) -> Iterator[dict[str, Any]]:
    """Read worker stdout and yield prefixed JSON events."""

    while True:
        line = process.stdout.readline() if process.stdout else ""
        if line:
            event = _parse_event_line(line)
            if event is not None:
                console_text = event.get("console")
                if console_text:
                    print(console_text, flush=True)
                yield event
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
                yield {"kind": "status", "message": line.rstrip()}
            continue
        if process.poll() is not None:
            break
        time.sleep(0.1)

    if process.stdout:
        for line in process.stdout.readlines():
            event = _parse_event_line(line)
            if event is not None:
                yield event
            else:
                sys.stdout.write(line)
                sys.stdout.flush()


def _parse_event_line(line: str) -> dict[str, Any] | None:
    """Parse one worker event line."""

    if not line.startswith(_EVENT_PREFIX):
        return None
    try:
        return json.loads(line[len(_EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None


def _wait_for_process_exit(process: subprocess.Popen) -> int | None:
    """Wait briefly for process exit after stdout closes."""

    try:
        return process.wait(timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        terminate_generation_process(
            process,
            timeout_seconds=_WORKER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        try:
            return process.wait(timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return process.poll()


def _read_result(path: Path, return_code: int | None) -> dict[str, Any]:
    """Read and validate the worker result JSON."""

    if not path.exists():
        raise RuntimeError("Isolated worker did not produce a result file.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if return_code != 0 or not result.get("success", False):
        raise RuntimeError(result.get("error") or f"Worker failed with code {return_code}")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with UTF-8 encoding."""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
