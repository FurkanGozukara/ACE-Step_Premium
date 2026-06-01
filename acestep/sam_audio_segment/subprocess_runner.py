"""Launch SAM-Audio processing in an isolated Python subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from acestep.core.generation.cancellation import (
    CANCEL_MESSAGE,
    GenerationCancelled,
    is_generation_cancelled,
    register_generation_subprocess,
    unregister_generation_subprocess,
)
from acestep.core.generation.subprocess_termination import terminate_generation_process
from .cancel import (
    check_sam_audio_cancelled,
    is_sam_audio_cancelled,
    register_sam_audio_subprocess,
    unregister_sam_audio_subprocess,
)
from .progress import ProgressCallback, parse_progress_line


def run_sam_audio_subprocess(
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a SAM-Audio request in a child process and return its JSON result."""

    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="acestep_sam_audio_") as temp_dir:
        check_sam_audio_cancelled()
        temp_root = Path(temp_dir)
        request_path = temp_root / "request.json"
        result_path = temp_root / "result.json"
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("ACESTEP_PROJECT_ROOT", str(project_root))
        command = [
            sys.executable,
            "-m",
            "acestep.sam_audio_segment.subprocess_worker",
            str(request_path),
            str(result_path),
        ]
        proc = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        register_generation_subprocess(proc)
        register_sam_audio_subprocess(proc)
        try:
            if is_sam_audio_cancelled():
                terminate_generation_process(proc)
            stdout, stderr = _collect_process_output(proc, progress_callback)
            return_code = proc.wait()
            was_cancelled = is_generation_cancelled() or is_sam_audio_cancelled()
        finally:
            unregister_sam_audio_subprocess(proc)
            unregister_generation_subprocess(proc)
        if was_cancelled:
            raise GenerationCancelled(CANCEL_MESSAGE)
        completed = subprocess.CompletedProcess(
            command,
            return_code,
            stdout=stdout,
            stderr=stderr,
        )
        if not result_path.is_file():
            raise RuntimeError(_format_subprocess_error(completed))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if return_code != 0 or not result.get("ok"):
            error = result.get("error") or _format_subprocess_error(completed)
            raise RuntimeError(error)
        return result


def _format_subprocess_error(proc: subprocess.CompletedProcess) -> str:
    """Return a compact subprocess failure message."""

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    detail = stderr or stdout or f"exit code {proc.returncode}"
    return f"SAM-Audio subprocess failed: {detail}"


def _collect_process_output(
    proc: subprocess.Popen,
    progress_callback: ProgressCallback | None,
) -> tuple[str, str]:
    """Collect subprocess output while forwarding progress lines."""

    output_queue: Queue[tuple[str, str]] = Queue()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = [
        Thread(target=_read_stream, args=(proc.stdout, "stdout", output_queue), daemon=True),
        Thread(target=_read_stream, args=(proc.stderr, "stderr", output_queue), daemon=True),
    ]
    for thread in threads:
        thread.start()
    while proc.poll() is None or not output_queue.empty() or any(
        thread.is_alive() for thread in threads
    ):
        if is_sam_audio_cancelled() and proc.poll() is None:
            terminate_generation_process(proc)
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except Empty:
            continue
        if stream_name == "stdout":
            progress = _handle_progress_line(line, progress_callback)
            if progress is None:
                stdout_lines.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
        else:
            stderr_lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()
    for thread in threads:
        thread.join(timeout=1.0)
    return "".join(stdout_lines), "".join(stderr_lines)


def _read_stream(stream, stream_name: str, output_queue: Queue[tuple[str, str]]) -> None:
    """Read one subprocess stream into a queue."""

    if stream is None:
        return
    for line in iter(stream.readline, ""):
        output_queue.put((stream_name, line))


def _handle_progress_line(
    line: str,
    progress_callback: ProgressCallback | None,
) -> tuple[float, str] | None:
    """Forward one progress line and return parsed data when handled."""

    progress = parse_progress_line(line.rstrip("\r\n"))
    if progress is None:
        return None
    fraction, message = progress
    if progress_callback is not None:
        progress_callback(fraction, message)
    return progress
