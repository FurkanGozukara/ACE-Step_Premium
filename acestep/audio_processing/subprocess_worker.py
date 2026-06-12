"""Audio Processing subprocess worker entry point."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from .auto_editor_workflow import export_auto_editor_workflow, workflow_export_enabled
from .file_processor import metrics_markdown, process_media_file
from .plots import make_spectrogram_figure
from .progress import ProgressCallback, encode_progress_line
from .settings import AudioProcessingSettings


def main(argv: list[str] | None = None) -> int:
    """Run one Audio Processing request from JSON and write a JSON result."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m acestep.audio_processing.subprocess_worker request result")
        return 2
    request_path = Path(args[0])
    result_path = Path(args[1])
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    settings = AudioProcessingSettings.from_payload(payload.get("settings"))
    try:
        if workflow_export_enabled(settings.workflow_export):
            result = _run_workflow(payload, settings, _stdout_progress)
        else:
            result = _run_media(payload, settings, _stdout_progress)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("ok") else 1


def _run_workflow(
    payload: dict,
    settings: AudioProcessingSettings,
    progress_callback: ProgressCallback,
) -> dict:
    """Export an Auto-Editor workflow-only file."""

    workflow_path = export_auto_editor_workflow(
        payload["input_path"],
        payload["output_dir"],
        payload["output_stem"],
        settings.workflow_export,
        settings.trim_settings(),
        process_callback=progress_callback,
        media_reference=payload.get("media_reference_path"),
    )
    return {
        "ok": True,
        "kind": "workflow",
        "input_path": payload["input_path"],
        "workflow_export": settings.workflow_export,
        "workflow_path": workflow_path,
        "media_reference_path": payload.get("media_reference_path"),
        "media_reference_is_local": bool(payload.get("media_reference_is_local")),
        "files": [workflow_path],
    }


def _run_media(
    payload: dict,
    settings: AudioProcessingSettings,
    progress_callback: ProgressCallback,
) -> dict:
    """Process one audio/video file and serialize UI-ready artifact paths."""

    result = process_media_file(
        payload["input_path"],
        payload["output_dir"],
        settings,
        progress_callback=progress_callback,
    )
    figure_path = payload.get("figure_path")
    if figure_path:
        _write_spectrogram_figure(result, Path(figure_path))
    return {
        "ok": True,
        "kind": "media",
        "audio_path": result.audio_path,
        "video_path": result.video_path,
        "metadata_path": result.metadata_path,
        "files": result.file_list(),
        "status_markdown": metrics_markdown(result),
        "figure_path": figure_path,
    }


def _write_spectrogram_figure(result, figure_path: Path) -> None:
    """Persist the matplotlib figure for the parent process to load."""

    figure = make_spectrogram_figure(
        result.processed_audio.before,
        result.processed_audio.after,
        result.processed_audio.sample_rate,
    )
    figure_path.write_bytes(pickle.dumps(figure))


def _stdout_progress(progress: float | None, message: str) -> None:
    """Write one progress event for the parent process to consume."""

    sys.stdout.write(encode_progress_line(progress, message) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
