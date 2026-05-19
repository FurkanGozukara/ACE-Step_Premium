"""Worker task implementations for isolated training UI subprocesses."""

from __future__ import annotations

import json
from typing import Any, Callable

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.training.dataset_builder import DatasetBuilder
from acestep.ui.gradio.events.training.dataset_ops import auto_label_all
from acestep.ui.gradio.events.training.lora_training import start_training
from acestep.ui.gradio.events.training.preprocess import preprocess_dataset
from acestep.ui.gradio.events.training.subprocess_safe_roots import apply_worker_safe_roots


Emit = Callable[[dict[str, Any]], None]
_SUCCESS = "\u2705"


def run_auto_label_task(payload: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Run dataset auto-labeling inside the worker process."""

    apply_worker_safe_roots(payload)
    builder = _load_builder(payload["dataset_path"])
    dit_handler = _new_dit_handler(payload.get("dit_init_params"))
    llm_handler = _new_llm_handler(payload.get("llm_init_params"))
    settings = dict(payload.get("settings") or {})
    progress = _worker_progress(emit)
    table, status, builder = auto_label_all(
        dit_handler,
        llm_handler,
        builder,
        bool(settings.get("skip_metas")),
        bool(settings.get("format_lyrics")),
        bool(settings.get("transcribe_lyrics")),
        str(settings.get("lm_lyrics_language") or "unknown"),
        bool(settings.get("only_unlabeled", True)),
        progress=progress,
        model_config=settings.get("model_config"),
        save_path=settings.get("save_path"),
        dataset_name=settings.get("dataset_name"),
        label_output_dir=settings.get("label_output_dir"),
        label_source_root=settings.get("label_source_root"),
    )
    _ = table
    dataset_path = payload["result_dataset_path"]
    save_status = builder.save_dataset(dataset_path, settings.get("dataset_name"))
    if not str(save_status).startswith(_SUCCESS):
        raise RuntimeError(save_status)
    status_text = _update_value(status, str(status))
    emit({"kind": "status", "message": status_text, "console": status_text})
    return {"success": True, "status": status_text, "dataset_path": dataset_path}


def run_preprocess_task(payload: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Run tensor preprocessing inside the worker process."""

    apply_worker_safe_roots(payload)
    builder = _load_builder(payload["dataset_path"])
    dit_handler = _new_dit_handler(payload.get("dit_init_params"))
    progress = _worker_progress(emit)
    status = preprocess_dataset(
        payload.get("output_dir"),
        payload.get("preprocess_mode"),
        dit_handler,
        builder,
        progress=progress,
        model_config=payload.get("model_config"),
    )
    status_text = str(status)
    emit({"kind": "status", "message": status_text, "console": status_text})
    return {"success": True, "status": status_text}


def run_lora_training_task(payload: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Run LoRA training inside the worker process."""

    dit_handler = _new_dit_handler(payload.get("dit_init_params"))
    args = dict(payload.get("training_args") or {})
    state = dict(payload.get("training_state") or {})
    last_status = "Training subprocess completed."
    last_log = ""
    for status, log_text, _plot, next_state in start_training(
        dit_handler=dit_handler,
        training_state=state,
        **args,
    ):
        state = _json_safe_state(next_state)
        last_status = str(status)
        last_log = str(log_text or last_log)
        emit(
            {
                "kind": "training",
                "status": last_status,
                "log": last_log,
                "training_state": state,
                "console": last_status,
            }
        )
    return {"success": True, "status": last_status, "log": last_log}


def _load_builder(dataset_path: str) -> DatasetBuilder:
    """Load a dataset JSON into a builder."""

    builder = DatasetBuilder()
    samples, status = builder.load_dataset(dataset_path)
    if not samples:
        raise RuntimeError(status)
    return builder


def _new_dit_handler(init_params: dict[str, Any] | None) -> AceStepHandler:
    """Create a DiT handler seeded with parent init parameters."""

    handler = AceStepHandler()
    handler.last_init_params = dict(init_params or {})
    return handler


def _new_llm_handler(init_params: dict[str, Any] | None) -> LLMHandler:
    """Create an LM handler seeded with parent init parameters."""

    handler = LLMHandler()
    handler.last_init_params = dict(init_params or {})
    return handler


def _worker_progress(emit: Emit) -> Callable[..., None]:
    """Build a Gradio-compatible progress callback that emits worker events."""

    def progress(value: Any = None, desc: Any = None, *args: Any, **kwargs: Any) -> None:
        _ = kwargs
        message = str(desc if desc is not None else _first_message(value, args))
        emit({"kind": "status", "message": message, "console": message})

    return progress


def _first_message(value: Any, args: tuple[Any, ...]) -> Any:
    """Return the first available progress message."""

    if isinstance(value, str):
        return value
    if args:
        return args[0]
    return value


def _update_value(value: Any, fallback: str) -> str:
    """Extract the payload from a Gradio update dictionary."""

    if isinstance(value, dict) and "value" in value:
        return str(value["value"])
    return fallback


def _json_safe_state(state: Any) -> dict[str, Any]:
    """Return a JSON-serializable training-state dictionary."""

    if not isinstance(state, dict):
        return {}
    try:
        json.dumps(state)
        return state
    except TypeError:
        return {
            key: value
            for key, value in state.items()
            if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
        }
