"""Helpers for optional isolated subprocess-based generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from acestep.ui.gradio.events.results.audio_playback_updates import build_audio_slot_update
from acestep.ui.gradio.events.results.batch_management_helpers import _extract_scores


_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".opus", ".aac"}


def _job_dir(project_root: str | Path) -> Path:
    target = Path(project_root).resolve() / ".cache" / "acestep" / "subprocess_jobs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_pending_core_outputs(status_text: str, is_format_caption: bool) -> tuple[Any, ...]:
    """Build the 46 core outputs for an in-flight subprocess status update."""
    return (
        *((gr.skip(),) * 8),
        gr.skip(),
        gr.skip(),
        status_text,
        gr.skip(),
        *((gr.skip(),) * 8),
        *((gr.skip(),) * 8),
        *((gr.skip(),) * 8),
        *((gr.skip(),) * 8),
        gr.skip(),
        is_format_caption,
    )


def build_final_core_outputs(result: dict[str, Any]) -> tuple[Any, ...]:
    """Build the 46 core UI outputs from a completed subprocess result."""
    audio_files = list(result.get("audio_files", []) or [])[:8]
    audio_updates = []
    for idx in range(8):
        path = audio_files[idx] if idx < len(audio_files) else None
        audio_updates.append(build_audio_slot_update(gr, path))

    scores = list(result.get("scores", []) or [])[:8]
    scores.extend([""] * (8 - len(scores)))
    codes = list(result.get("codes", []) or [])[:8]
    codes.extend([""] * (8 - len(codes)))
    lrcs = list(result.get("lrcs", []) or [])[:8]
    lrcs.extend([""] * (8 - len(lrcs)))

    return (
        *audio_updates,
        result.get("all_audio_paths"),
        result.get("generation_info", ""),
        result.get("status_output", "Generation Complete"),
        result.get("seed_value", ""),
        *scores,
        *codes,
        *((gr.skip(),) * 8),
        *lrcs,
        result.get("lm_metadata"),
        bool(result.get("is_format_caption", False)),
    )


def stream_subprocess_generation(request_payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Launch the worker process and yield status updates followed by the final result."""
    project_root = str(request_payload["project_root"])
    work_dir = _job_dir(project_root)
    job_id = uuid.uuid4().hex
    request_path = work_dir / f"{job_id}.request.json"
    result_path = work_dir / f"{job_id}.result.json"

    with request_path.open("w", encoding="utf-8") as handle:
        json.dump(request_payload, handle, indent=2, ensure_ascii=False)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [
        sys.executable,
        "-m",
        "acestep.ui.gradio.events.results.subprocess_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    log_lines: list[str] = []
    yield {"kind": "status", "message": "Starting isolated generation worker..."}

    try:
        while True:
            line = process.stdout.readline() if process.stdout else ""
            if line:
                log_lines.append(line.rstrip())
                yield {"kind": "status", "message": "\n".join(log_lines[-18:])}
                continue

            if process.poll() is not None:
                break
            time.sleep(0.1)

        if process.stdout:
            for remaining in process.stdout.readlines():
                log_lines.append(remaining.rstrip())
    finally:
        return_code = process.wait()

    if not result_path.exists():
        raise RuntimeError(
            "Subprocess worker did not produce a result file.\n"
            + "\n".join(log_lines[-20:])
        )

    with result_path.open("r", encoding="utf-8") as handle:
        result_data = json.load(handle)

    if return_code != 0 or not result_data.get("success", False):
        raise RuntimeError(
            result_data.get("error")
            or f"Subprocess generation failed with exit code {return_code}"
        )

    all_audio_paths = list(result_data.get("all_audio_paths", []) or [])
    audio_files = [
        path for path in all_audio_paths
        if Path(str(path)).suffix.lower() in _AUDIO_SUFFIXES
    ]
    result_data["audio_files"] = audio_files
    result_data["status_output"] = "\n".join(log_lines[-18:]) or result_data.get(
        "status_output", "Generation Complete"
    )
    yield {"kind": "result", "result": result_data}
