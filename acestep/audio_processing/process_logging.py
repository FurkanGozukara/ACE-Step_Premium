"""Command logging helpers for Audio Processing external tools."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from loguru import logger

from .process_streaming import stream_command_output


ProcessCallback = Callable[..., None]


def emit_process_message(
    process_callback: ProcessCallback | None,
    message: str,
    progress_value: float | None = None,
) -> None:
    """Emit one process status line to the terminal and optional UI callback."""

    text = str(message or "").strip()
    if not text:
        return
    logger.info("[audio_processing] {}", text)
    if process_callback is None:
        return
    try:
        process_callback(progress_value, text)
    except TypeError:
        process_callback(text)


def run_external_command(
    cmd: list[str],
    message: str,
    *,
    process_callback: ProcessCallback | None = None,
    progress_duration_seconds: float | None = None,
    timeout: int = 1800,
) -> None:
    """Run an external command and mirror progress to terminal and UI callbacks."""

    display_message = message.removesuffix(" failed")
    emit_process_message(process_callback, f"Starting: {display_message}")
    emit_process_message(process_callback, f"Command: {subprocess.list2cmdline(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{message}: executable not found.") from exc

    try:
        output = stream_command_output(
            process,
            display_message,
            process_callback,
            progress_duration_seconds,
            timeout,
            emit_process_message,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{message}: timed out.") from exc

    return_code = process.wait()
    if return_code != 0:
        stderr = output.strip() or f"exit code {return_code}"
        raise RuntimeError(f"{message}: {stderr}")
    emit_process_message(process_callback, f"Finished: {display_message}")
