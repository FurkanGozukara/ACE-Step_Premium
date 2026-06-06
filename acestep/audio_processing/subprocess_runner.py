"""Launch Audio Processing in an isolated Python subprocess."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from .cancel import (
    AudioProcessingCancelled,
    is_audio_processing_cancelled,
    register_audio_processing_subprocess,
    terminate_audio_processing_process,
    unregister_audio_processing_subprocess,
)
from .progress import ProgressCallback, parse_progress_line


def run_audio_processing_subprocess(
    payload: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run an Audio Processing request in a child process and return its result."""

    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="acestep_audio_processing_") as temp_dir:
        temp_root = Path(temp_dir)
        request_path = temp_root / "request.json"
        result_path = temp_root / "result.json"
        payload = dict(payload)
        payload["figure_path"] = str(temp_root / "spectrogram.pkl")
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        proc = _start_worker(project_root, request_path, result_path)
        register_audio_processing_subprocess(proc)
        try:
            if is_audio_processing_cancelled():
                terminate_audio_processing_process(proc)
            stdout, stderr = _collect_process_output(proc, progress_callback)
            return_code = proc.wait()
            was_cancelled = is_audio_processing_cancelled()
        finally:
            unregister_audio_processing_subprocess(proc)
        if was_cancelled:
            raise AudioProcessingCancelled("Audio Processing cancelled.")
        return _read_result(result_path, return_code, stdout, stderr)


def _start_worker(
    project_root: Path,
    request_path: Path,
    result_path: Path,
) -> subprocess.Popen:
    """Start the Audio Processing worker process."""

    env = os.environ.copy()
    env.setdefault("ACESTEP_PROJECT_ROOT", str(project_root))
    command = [
        sys.executable,
        "-m",
        "acestep.audio_processing.subprocess_worker",
        str(request_path),
        str(result_path),
    ]
    return subprocess.Popen(
        command,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


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
        if is_audio_processing_cancelled() and proc.poll() is None:
            terminate_audio_processing_process(proc)
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except Empty:
            continue
        if stream_name == "stdout":
            if _handle_progress_line(line, progress_callback):
                continue
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
    stream.close()


def _handle_progress_line(
    line: str,
    progress_callback: ProgressCallback | None,
) -> bool:
    """Forward one progress line and return whether it was handled."""

    progress = parse_progress_line(line.rstrip("\r\n"))
    if progress is None:
        return False
    fraction, message = progress
    if progress_callback is not None:
        progress_callback(fraction, message)
    return True


def _read_result(
    result_path: Path,
    return_code: int,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    """Read the worker JSON result and attach the in-memory spectrogram figure."""

    if not result_path.is_file():
        raise RuntimeError(_format_subprocess_error(return_code, stdout, stderr))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if return_code != 0 or not result.get("ok"):
        raise RuntimeError(result.get("error") or _format_subprocess_error(
            return_code, stdout, stderr
        ))
    figure_path = Path(str(result.get("figure_path") or ""))
    if figure_path.is_file():
        result["figure"] = pickle.loads(figure_path.read_bytes())
    return result


def _format_subprocess_error(return_code: int, stdout: str, stderr: str) -> str:
    """Return a compact subprocess failure message."""

    detail = (stderr or stdout).strip() or f"exit code {return_code}"
    return f"Audio Processing subprocess failed: {detail}"
