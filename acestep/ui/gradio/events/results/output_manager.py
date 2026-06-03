"""Helpers for premium generation output folders and metadata manifests."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acestep.ui.gradio.events.results.output_paths import (
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    create_generation_run_dir,
    get_active_generation_run_name,
    get_active_results_dir,
    get_results_dir,
    use_generation_run_name,
    use_results_dir,
)
from acestep.ui.gradio.media_upload_values import latest_upload_path

__all__ = [
    "DEFAULT_RESULTS_DIR",
    "PROJECT_ROOT",
    "build_generation_manifest",
    "create_generation_run_dir",
    "get_active_generation_run_name",
    "get_active_results_dir",
    "get_results_dir",
    "make_json_safe",
    "persist_generation_inputs",
    "use_generation_run_name",
    "use_results_dir",
    "write_json",
    "write_text",
]


def make_json_safe(value: Any) -> Any:
    """Convert nested data to JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    """Write JSON with UTF-8 and stable indentation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return str(target.resolve()).replace("\\", "/")


def write_text(path: str | Path, content: Any) -> str:
    """Write a UTF-8 text file and return its normalized absolute path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("" if content is None else str(content))
    return str(target.resolve()).replace("\\", "/")


def _copy_run_asset(
    *,
    run_dir: str | Path,
    source_path: Any,
    target_stem: str,
) -> str | None:
    """Copy an uploaded source asset into the run folder when available."""
    raw_source = latest_upload_path(source_path) or ""
    if not raw_source:
        return None

    source = Path(raw_source).expanduser()
    try:
        source = source.resolve()
    except Exception:
        pass

    if not source.exists() or not source.is_file():
        return None

    suffix = source.suffix or ".bin"
    target = Path(run_dir) / f"{target_stem}{suffix}"
    shutil.copy2(source, target)
    return str(target.resolve()).replace("\\", "/")


def persist_generation_inputs(
    *,
    run_dir: str | Path,
    caption: Any,
    lyrics: Any,
    reference_audio: Any,
    src_audio: Any,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the user inputs and a full request snapshot inside the run folder."""
    run_path = Path(run_dir)
    caption_path = write_text(run_path / "caption.txt", caption)
    lyrics_path = write_text(run_path / "lyrics.txt", lyrics)
    reference_audio_path = _copy_run_asset(
        run_dir=run_path,
        source_path=reference_audio,
        target_stem="reference_audio",
    )
    source_audio_path = _copy_run_asset(
        run_dir=run_path,
        source_path=src_audio,
        target_stem="source_audio",
    )

    assets = {
        "caption_path": caption_path,
        "lyrics_path": lyrics_path,
        "reference_audio_path": reference_audio_path,
        "source_audio_path": source_audio_path,
        "original_reference_audio": latest_upload_path(reference_audio),
        "original_source_audio": latest_upload_path(src_audio),
    }
    request_path = write_json(
        run_path / "generation_request.json",
        {
            "_meta": {
                "format": "ace_step_generation_request",
                "version": 1,
                "run_dir": str(run_path.resolve()).replace("\\", "/"),
            },
            "request": make_json_safe(request_payload or {}),
            "assets": make_json_safe(assets),
        },
    )
    assets["request_path"] = request_path
    return assets


def build_generation_manifest(
    *,
    run_dir: str | Path,
    request_started_at: float,
    request_finished_at: float | None = None,
    generation_info: str = "",
    seed_value: str = "",
    audio_format: str = "",
    sample_files: list[dict[str, Any]] | None = None,
    time_costs: dict[str, Any] | None = None,
    request_payload: dict[str, Any] | None = None,
    lm_metadata: dict[str, Any] | None = None,
    status: str = "completed",
) -> str:
    """Write the top-level generation manifest for a run folder."""
    finished_at = request_finished_at or request_started_at
    payload = {
        "_meta": {
            "format": "ace_step_generation_manifest",
            "version": 1,
            "status": status,
            "run_dir": str(Path(run_dir).resolve()).replace("\\", "/"),
            "started_at_utc": datetime.fromtimestamp(
                request_started_at, tz=timezone.utc
            ).isoformat(),
            "finished_at_utc": datetime.fromtimestamp(
                finished_at, tz=timezone.utc
            ).isoformat(),
            "duration_seconds": round(max(0.0, finished_at - request_started_at), 3),
        },
        "generation_info": generation_info,
        "seed_value": seed_value,
        "audio_format": audio_format,
        "time_costs": make_json_safe(time_costs or {}),
        "request": make_json_safe(request_payload or {}),
        "lm_metadata": make_json_safe(lm_metadata or {}),
        "samples": make_json_safe(sample_files or []),
    }
    return write_json(Path(run_dir) / "generation_manifest.json", payload)
