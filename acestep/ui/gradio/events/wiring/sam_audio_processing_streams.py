"""Streaming wrappers for SAM-Audio Gradio subprocess work."""

from __future__ import annotations

import time
from threading import Thread
from typing import Any, Callable

import gradio as gr

from acestep.core.generation.cancellation import GenerationCancelled
from acestep.sam_audio_segment.cancel import (
    request_sam_audio_cancel,
    sam_audio_cancel_scope,
)

from .sam_audio_action_helpers import single_status
from .sam_audio_status_log import SamAudioStatusLog

SingleWorker = Callable[[], tuple[dict[str, Any], list[str]]]
BatchWorker = Callable[[], dict[str, Any]]


def stream_single_subprocess(
    worker: SingleWorker,
    cleanup_status: str,
    status_log: SamAudioStatusLog,
):
    """Stream one SAM-Audio subprocess worker into Gradio single-file outputs."""

    holder = _start_worker(worker)
    try:
        yield _single_pending_update(
            status_log.render(_with_cleanup(cleanup_status, "SAM-Audio started."))
        )
        yield from _yield_pending_single_while_running(holder, cleanup_status, status_log)
    finally:
        _cancel_active_worker(holder)
    error = holder.get("error")
    if isinstance(error, GenerationCancelled):
        yield _single_pending_update(status_log.append_to_status("SAM-Audio cancelled."))
        return
    if error is not None:
        yield _single_pending_update(status_log.append_to_status(f"SAM-Audio failed: {error}"))
        return
    artifacts, files = holder["result"]
    status = status_log.append_to_status(single_status(artifacts, cleanup_status))
    yield (
        artifacts.get("target_audio_path"),
        artifacts.get("residual_audio_path"),
        gr.update(
            value=artifacts.get("target_video_path"),
            visible=bool(artifacts.get("target_video_path")),
        ),
        gr.update(value=files, visible=True),
        status,
    )


def stream_batch_subprocess(
    worker: BatchWorker,
    cleanup_status: str,
    status_log: SamAudioStatusLog,
):
    """Stream one SAM-Audio subprocess worker into Gradio batch outputs."""

    holder = _start_worker(worker)
    try:
        yield status_log.render(
            _with_cleanup(cleanup_status, "SAM-Audio batch started.")
        ), gr.update(visible=False)
        while holder["thread"].is_alive():
            if status_log.drain():
                yield status_log.render(
                    _with_cleanup(cleanup_status, "SAM-Audio batch running...")
                ), gr.update(visible=False)
            time.sleep(0.25)
        holder["thread"].join()
        status_log.drain()
    finally:
        _cancel_active_worker(holder)
    error = holder.get("error")
    if isinstance(error, GenerationCancelled):
        yield status_log.append_to_status("SAM-Audio batch cancelled."), gr.update(
            visible=False
        )
        return
    if error is not None:
        yield status_log.append_to_status(f"SAM-Audio batch failed: {error}"), gr.update(
            visible=False
        )
        return
    result = holder["result"]
    yield status_log.append_to_status(result.get("status", "Batch complete.")), gr.update(
        value=result.get("files", []),
        visible=bool(result.get("files")),
    )


def _start_worker(worker: Callable[[], Any]) -> dict[str, Any]:
    """Start a daemon worker thread and return its shared result holder."""

    holder: dict[str, Any] = {}

    def _run() -> None:
        try:
            with sam_audio_cancel_scope():
                holder["result"] = worker()
        except BaseException as exc:
            holder["error"] = exc

    holder["thread"] = Thread(target=_run, daemon=True)
    holder["thread"].start()
    return holder


def _yield_pending_single_while_running(
    holder: dict[str, Any],
    cleanup_status: str,
    status_log: SamAudioStatusLog,
):
    """Yield single-file pending updates while a worker thread runs."""

    while holder["thread"].is_alive():
        if status_log.drain():
            yield _single_pending_update(
                status_log.render(_with_cleanup(cleanup_status, "SAM-Audio running..."))
            )
        time.sleep(0.25)
    holder["thread"].join()
    status_log.drain()


def _single_pending_update(status: str) -> tuple[Any, ...]:
    """Return output placeholders that only update the status panel."""

    return gr.update(), gr.update(), gr.update(), gr.update(visible=False), status


def _cancel_active_worker(holder: dict[str, Any]) -> None:
    """Cancel and briefly join an active SAM-Audio worker."""

    thread = holder.get("thread")
    if thread is None or not thread.is_alive():
        return
    request_sam_audio_cancel()
    thread.join(timeout=3.0)


def _with_cleanup(cleanup_status: str, status: str) -> str:
    """Prefix status with generation cleanup details when present."""

    return f"{cleanup_status}\n{status}" if cleanup_status else status
