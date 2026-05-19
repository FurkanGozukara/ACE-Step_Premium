"""Subprocess-backed auto-label and preprocess handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from acestep.training.dataset_builder import DatasetBuilder

from .subprocess_runner import (
    TrainingSubprocessJob,
    create_training_subprocess_job,
    stream_training_subprocess_job,
)
from .subprocess_safe_roots import build_worker_safe_roots


_SUCCESS = "\u2705"


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
    dataset_path = _save_request_dataset(builder_state, settings.get("dataset_name"), job)
    result_dataset_path = job.work_dir / "auto_label_result.json"
    safe_roots = build_worker_safe_roots(
        dit_init_params["project_root"],
        dataset_path,
        result_dataset_path,
        settings.get("save_path"),
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
                builder = _load_result_builder(result["dataset_path"])
                return (
                    gr.update(value=builder.get_samples_dataframe_data()),
                    gr.update(value=result.get("status") or status),
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


def _report_progress(progress: Any, message: str) -> None:
    """Forward subprocess status to an optional Gradio progress object."""

    if progress is None:
        return
    try:
        progress(None, desc=message)
    except TypeError:
        progress(message)
