"""Subprocess-backed auto-label and preprocess handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from loguru import logger

from acestep.training.dataset_builder import DatasetBuilder

from .auto_label_control import (
    has_active_inline_auto_label,
    request_auto_label_cancel,
)
from .subprocess_control import request_training_subprocess_stop
from .subprocess_runner import (
    TrainingSubprocessJob,
    create_training_subprocess_job,
    stream_training_subprocess_job,
)
from .subprocess_safe_roots import build_worker_safe_roots


_SUCCESS = "\u2705"
AUTO_LABEL_CANCEL_CONFIRM_JS = (
    "() => { if (!confirm('Are you sure you want to cancel the current "
    "auto-label run?')) { throw new Error('Auto-label cancel aborted.'); } "
    "return []; }"
)
AUTO_LABEL_CANCEL_REQUESTED_STATUS = (
    "Stopping isolated auto-label subprocess..."
)
NO_ACTIVE_AUTO_LABEL_STATUS = "No active isolated auto-label subprocess is currently running."
INLINE_AUTO_LABEL_CANCEL_REQUESTED_STATUS = (
    "Auto-label cancellation requested. In-process labeling will stop after the "
    "current file or batch step."
)
_GENERIC_STOPPED_STATUS = "Isolated training subprocess stopped."
_AUTO_LABEL_STOPPED_STATUS = "Isolated auto-label subprocess stopped."


def request_auto_label_cancel_from_ui(confirmed: bool = True) -> str | Any:
    """Request cancellation for the active isolated auto-label worker."""

    if not confirmed:
        return gr.skip()

    request_auto_label_cancel()
    stopped_subprocess = request_training_subprocess_stop()
    if not stopped_subprocess:
        if has_active_inline_auto_label():
            logger.info("In-process auto-label cancellation requested from UI.")
            return INLINE_AUTO_LABEL_CANCEL_REQUESTED_STATUS
        logger.info("Auto-label cancel requested from UI, but no subprocess is active.")
        return NO_ACTIVE_AUTO_LABEL_STATUS

    logger.info("Auto-label subprocess cancellation requested from UI.")
    return AUTO_LABEL_CANCEL_REQUESTED_STATUS


def run_auto_label_subprocess(
    *,
    builder_state: DatasetBuilder | None,
    settings: dict[str, Any],
    dit_init_params: dict[str, Any],
    llm_init_params: dict[str, Any],
    progress: Any = None,
) -> tuple[Any, Any, DatasetBuilder | None]:
    """Run dataset auto-labeling in a worker process and return UI updates."""

    if builder_state is None or not getattr(builder_state, "samples", None):
        return [], "No samples to label. Please scan a directory first.", builder_state

    job = create_training_subprocess_job(dit_init_params["project_root"])
    settings = dict(settings)
    settings["label_source_root"] = getattr(builder_state, "_current_dir", None)
    dataset_path = _save_request_dataset(builder_state, settings.get("dataset_name"), job)
    result_dataset_path = job.work_dir / "auto_label_result.json"
    safe_roots = build_worker_safe_roots(
        dit_init_params["project_root"],
        dataset_path,
        result_dataset_path,
        settings.get("save_path"),
        settings.get("label_output_dir"),
        settings.get("label_source_root"),
        samples=builder_state.samples,
    )
    payload = {
        "operation": "auto_label",
        "project_root": dit_init_params["project_root"],
        "dit_init_params": dit_init_params,
        "llm_init_params": llm_init_params,
        "dataset_path": str(dataset_path),
        "result_dataset_path": str(result_dataset_path),
        "safe_roots": safe_roots,
        "settings": settings,
    }
    status = "Starting isolated auto-label worker..."
    try:
        for event in stream_training_subprocess_job(payload, job):
            if event.get("kind") == "status":
                status = str(event.get("message") or status)
                _report_progress(progress, status)
            elif event.get("kind") == "result":
                result = event["result"]
                if not result.get("dataset_path"):
                    status = _auto_label_status_from_result(result, status)
                    return (
                        gr.update(value=builder_state.get_samples_dataframe_data()),
                        gr.update(value=status),
                        builder_state,
                    )
                builder = _load_result_builder(result["dataset_path"])
                return (
                    gr.update(value=builder.get_samples_dataframe_data()),
                    gr.update(value=_auto_label_status_from_result(result, status)),
                    builder,
                )
    except Exception as exc:
        status = f"Auto-label subprocess failed: {exc!s}"
    return (
        gr.update(value=builder_state.get_samples_dataframe_data()),
        gr.update(value=status),
        builder_state,
    )


def run_preprocess_subprocess(
    *,
    output_dir: str,
    preprocess_mode: str,
    builder_state: DatasetBuilder | None,
    model_config: str | None,
    dit_init_params: dict[str, Any],
    progress: Any = None,
) -> str:
    """Run tensor preprocessing in a worker process and return its final status."""

    if builder_state is None or not getattr(builder_state, "samples", None):
        return "No dataset loaded. Please scan a directory first."

    job = create_training_subprocess_job(dit_init_params["project_root"])
    dataset_path = _save_request_dataset(builder_state, None, job)
    safe_roots = build_worker_safe_roots(
        dit_init_params["project_root"],
        dataset_path,
        output_dir,
        samples=builder_state.samples,
    )
    payload = {
        "operation": "preprocess",
        "project_root": dit_init_params["project_root"],
        "dit_init_params": dit_init_params,
        "dataset_path": str(dataset_path),
        "output_dir": output_dir,
        "preprocess_mode": preprocess_mode,
        "model_config": model_config,
        "safe_roots": safe_roots,
    }
    status = "Starting isolated preprocess worker..."
    try:
        for event in stream_training_subprocess_job(payload, job):
            if event.get("kind") == "status":
                status = str(event.get("message") or status)
                _report_progress(progress, status)
            elif event.get("kind") == "result":
                return str(event["result"].get("status") or status)
    except Exception as exc:
        return f"Preprocess subprocess failed: {exc!s}"
    return status


def _save_request_dataset(
    builder_state: DatasetBuilder,
    dataset_name: str | None,
    job: TrainingSubprocessJob,
) -> Path:
    """Save the current builder state for a worker request."""

    dataset_path = job.work_dir / "dataset_request.json"
    status = builder_state.save_dataset(str(dataset_path), dataset_name)
    if not str(status).startswith(_SUCCESS):
        raise RuntimeError(status)
    return dataset_path


def _load_result_builder(dataset_path: str) -> DatasetBuilder:
    """Load a worker-written dataset into a new builder instance."""

    builder = DatasetBuilder()
    samples, status = builder.load_dataset(dataset_path)
    if not samples:
        raise RuntimeError(status)
    return builder


def _auto_label_status_from_result(result: dict[str, Any], fallback: str) -> str:
    """Return auto-label wording for a subprocess result status."""

    status = str(result.get("status") or fallback)
    if status == _GENERIC_STOPPED_STATUS:
        return _AUTO_LABEL_STOPPED_STATUS
    return status


def _report_progress(progress: Any, message: str) -> None:
    """Forward subprocess status to an optional Gradio progress object."""

    if progress is None:
        return
    try:
        progress(None, desc=message)
    except TypeError:
        progress(message)
