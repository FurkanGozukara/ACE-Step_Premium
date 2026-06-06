"""Streaming subprocess output parsing for Audio Processing tools."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from queue import Empty, Queue
from threading import Thread
from time import monotonic


ProcessCallback = Callable[..., None]
ProcessEmitter = Callable[[ProcessCallback | None, str, float | None], None]
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PERCENT_RE = re.compile(r"(?<!\d)(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)%")
_FFMPEG_PROGRESS_KEYS = {
    "bitrate",
    "drop_frames",
    "dup_frames",
    "fps",
    "frame",
    "out_time",
    "out_time_ms",
    "out_time_us",
    "progress",
    "speed",
    "stream_0_0_q",
    "total_size",
}


def stream_command_output(
    process: subprocess.Popen[str],
    display_message: str,
    process_callback: ProcessCallback | None,
    progress_duration_seconds: float | None,
    timeout: int,
    emit_message: ProcessEmitter,
) -> str:
    """Stream command output, returning cleaned lines for failure messages."""

    output_queue: Queue[str | None] = Queue()
    reader = Thread(target=_enqueue_output, args=(process, output_queue), daemon=True)
    reader.start()
    start = monotonic()
    buffer: list[str] = []
    output_lines: list[str] = []
    reader_done = False

    while True:
        if timeout > 0 and monotonic() - start > timeout:
            _terminate_process(process)
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            item = output_queue.get(timeout=0.1)
        except Empty:
            if reader_done and process.poll() is not None:
                break
            continue
        if item is None:
            reader_done = True
            if process.poll() is not None and output_queue.empty():
                break
            continue
        if item in {"\r", "\n"}:
            _flush_output_buffer(
                buffer,
                output_lines,
                display_message,
                process_callback,
                progress_duration_seconds,
                emit_message,
            )
            continue
        buffer.append(item)

    _flush_output_buffer(
        buffer,
        output_lines,
        display_message,
        process_callback,
        progress_duration_seconds,
        emit_message,
    )
    return "\n".join(output_lines)


def _enqueue_output(process: subprocess.Popen[str], output_queue: Queue[str | None]) -> None:
    """Read subprocess output one character at a time for carriage-return progress."""

    try:
        if process.stdout is None:
            return
        while True:
            char = process.stdout.read(1)
            if char == "":
                break
            output_queue.put(char)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        output_queue.put(None)


def _flush_output_buffer(
    buffer: list[str],
    output_lines: list[str],
    display_message: str,
    process_callback: ProcessCallback | None,
    progress_duration_seconds: float | None,
    emit_message: ProcessEmitter,
) -> None:
    """Emit one buffered output line after cleaning and progress parsing."""

    if not buffer:
        return
    line = _clean_process_output("".join(buffer))
    buffer.clear()
    if not line:
        return
    emitted = _emit_progress_line(
        line,
        display_message,
        process_callback,
        progress_duration_seconds,
        emit_message,
    )
    if emitted:
        output_lines.append(emitted)


def _emit_progress_line(
    line: str,
    display_message: str,
    process_callback: ProcessCallback | None,
    progress_duration_seconds: float | None,
    emit_message: ProcessEmitter,
) -> str | None:
    """Emit a process output line and return text useful for failure messages."""

    key, value = _split_progress_key_value(line)
    if key in _FFMPEG_PROGRESS_KEYS:
        progress = _ffmpeg_progress_fraction(key, value, progress_duration_seconds)
        if progress is not None:
            text = f"{display_message}: {_format_percent(progress)}"
            emit_message(process_callback, text, progress)
            return text
        return None

    progress = _percent_fraction(line)
    emit_message(process_callback, line, progress)
    return line


def _clean_process_output(text: str) -> str:
    """Return process output without ANSI terminal controls."""

    return _ANSI_ESCAPE_RE.sub("", text).replace("\b", "").strip()


def _split_progress_key_value(line: str) -> tuple[str, str]:
    """Split an ffmpeg progress key-value line."""

    key, separator, value = line.partition("=")
    if not separator:
        return "", ""
    return key.strip(), value.strip()


def _ffmpeg_progress_fraction(
    key: str,
    value: str,
    progress_duration_seconds: float | None,
) -> float | None:
    """Return progress fraction from ffmpeg progress output when duration is known."""

    if key == "progress" and value == "end":
        return 1.0
    if key != "out_time_ms" or not progress_duration_seconds:
        return None
    try:
        time_value = float(value) / 1_000_000.0
    except ValueError:
        return None
    return max(0.0, min(0.999, time_value / progress_duration_seconds))


def _percent_fraction(line: str) -> float | None:
    """Return a 0..1 fraction parsed from text that contains a percentage."""

    match = _PERCENT_RE.search(line)
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(1)) / 100.0))
    except ValueError:
        return None


def _format_percent(progress: float) -> str:
    """Return progress as a one-decimal percentage string."""

    return f"{max(0.0, min(1.0, progress)) * 100:.1f}%"


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate a process that exceeded its timeout."""

    process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
