"""CLI worker for isolated training UI actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from acestep.ui.gradio.events.training.subprocess_worker_tasks import (
    run_auto_label_task,
    run_lora_training_task,
    run_preprocess_task,
)


_EVENT_PREFIX = "ACE_TRAINING_EVENT "


def main() -> int:
    """Run one isolated training UI worker request."""

    parser = argparse.ArgumentParser(description="ACE-Step isolated training worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    try:
        payload = _read_json(args.request)
        project_root = str(Path(payload["project_root"]).resolve())
        os.environ.setdefault("ACESTEP_PROJECT_ROOT", project_root)
        operation = str(payload.get("operation") or "")
        _emit({"kind": "status", "message": f"Worker starting: {operation}"})
        result = _dispatch(payload)
        _write_json(args.result, result)
        _emit({"kind": "status", "message": f"Worker completed: {operation}"})
        return 0
    except Exception as exc:
        _write_json(args.result, {"success": False, "error": str(exc)})
        _emit({"kind": "status", "message": f"Worker error: {exc}"})
        return 1


def _dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a worker payload by operation name."""

    operation = str(payload.get("operation") or "")
    if operation == "auto_label":
        return run_auto_label_task(payload, _emit)
    if operation == "preprocess":
        return run_preprocess_task(payload, _emit)
    if operation == "lora_training":
        return run_lora_training_task(payload, _emit)
    raise RuntimeError(f"Unknown worker operation: {operation}")


def _emit(event: dict[str, Any]) -> None:
    """Write one parent-readable worker event."""

    print(_EVENT_PREFIX + json.dumps(event, ensure_ascii=False), flush=True)


def _read_json(path: str) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
