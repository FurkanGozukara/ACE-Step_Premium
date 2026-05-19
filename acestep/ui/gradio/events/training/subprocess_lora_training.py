"""Subprocess-backed LoRA training stream helpers."""

from __future__ import annotations

import re
from typing import Any, Iterator

from .subprocess_runner import (
    create_training_subprocess_job,
    stream_training_subprocess_job,
)
from .training_utils import _training_loss_figure


_STEP_LOSS_RE = re.compile(r"Step\s+(\d+).*?Loss:\s+([-+]?\d+(?:\.\d+)?)")


def stream_lora_training_subprocess(
    *,
    dit_init_params: dict[str, Any],
    training_args: dict[str, Any],
    training_state: dict[str, Any],
) -> Iterator[tuple[Any, Any, Any, dict[str, Any]]]:
    """Run LoRA training in an isolated worker and stream Gradio outputs."""

    state = dict(training_state)
    state["is_training"] = True
    state["should_stop"] = False
    job = create_training_subprocess_job(dit_init_params["project_root"])
    payload = {
        "operation": "lora_training",
        "project_root": dit_init_params["project_root"],
        "dit_init_params": dit_init_params,
        "training_args": training_args,
        "training_state": state,
    }
    step_list: list[int] = []
    loss_list: list[float] = []
    log_text = ""

    for event in stream_training_subprocess_job(payload, job):
        kind = event.get("kind")
        if kind == "status":
            status = str(event.get("message") or "")
            plot = _training_loss_figure(state, step_list, loss_list)
            yield status, log_text, plot, state
            continue
        if kind == "training":
            status = str(event.get("status") or "")
            log_text = str(event.get("log") or log_text)
            state = _event_state(event, state)
            _append_loss_point(status, step_list, loss_list)
            plot = _training_loss_figure(state, step_list, loss_list)
            yield status, log_text, plot, state
            continue
        if kind == "result":
            result = event["result"]
            state["is_training"] = False
            state["should_stop"] = False
            status = str(result.get("status") or "Training subprocess completed.")
            log_text = str(result.get("log") or log_text)
            plot = _training_loss_figure(state, step_list, loss_list)
            yield status, log_text, plot, state


def _event_state(event: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe training state from a worker event."""

    event_state = event.get("training_state")
    if isinstance(event_state, dict):
        return event_state
    return fallback


def _append_loss_point(
    status: str,
    step_list: list[int],
    loss_list: list[float],
) -> None:
    """Extract a step/loss point from status text when available."""

    match = _STEP_LOSS_RE.search(status)
    if not match:
        return
    step = int(match.group(1))
    if step_list and step <= step_list[-1]:
        return
    step_list.append(step)
    loss_list.append(float(match.group(2)))
