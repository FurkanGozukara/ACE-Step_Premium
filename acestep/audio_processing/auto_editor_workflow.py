"""Auto-Editor workflow export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .auto_editor_runner import auto_editor_command, run_command
from .auto_editor_trim_settings import AutoEditorTrimSettings
from .auto_editor_workflow_paths import rewrite_fcpxml_media_references
from .process_logging import ProcessCallback, emit_process_message


AUTO_EDITOR_WORKFLOW_EXPORT_NONE = "none"
AUTO_EDITOR_WORKFLOW_EXPORT_KEY = "ap_auto_editor_workflow_export"
AUTO_EDITOR_WORKFLOW_EXPORT_CHOICES: tuple[tuple[str, str], ...] = (
    ("None", AUTO_EDITOR_WORKFLOW_EXPORT_NONE),
    ("DaVinci Resolve", "resolve"),
    ("DaVinci Resolve FCP7 XML", "resolve-fcp7"),
    ("Adobe Premiere Pro", "premiere"),
    ("Adobe Premiere Pro OTIO", "premiere-otio"),
    ("Final Cut Pro", "final-cut-pro"),
    ("Shotcut", "shotcut"),
    ("Kdenlive", "kdenlive"),
)
_WORKFLOW_EXPORT_EXTENSIONS: dict[str, str] = {
    "resolve": "fcpxml",
    "resolve-fcp7": "xml",
    "premiere": "xml",
    "premiere-otio": "otio",
    "final-cut-pro": "fcpxml",
    "shotcut": "mlt",
    "kdenlive": "kdenlive",
}


def normalize_workflow_export_mode(value: Any) -> str:
    """Return a supported Auto-Editor workflow export mode."""

    mode = str(value or AUTO_EDITOR_WORKFLOW_EXPORT_NONE).strip()
    return mode if mode in _WORKFLOW_EXPORT_EXTENSIONS else AUTO_EDITOR_WORKFLOW_EXPORT_NONE


def workflow_export_enabled(mode: Any) -> bool:
    """Return whether a workflow export mode is selected."""

    return normalize_workflow_export_mode(mode) != AUTO_EDITOR_WORKFLOW_EXPORT_NONE


def export_auto_editor_workflow(
    source_media: str | Path,
    output_dir: str | Path,
    output_stem: str,
    mode: str,
    trim_settings: AutoEditorTrimSettings,
    process_callback: ProcessCallback | None = None,
    media_reference: str | Path | None = None,
) -> str:
    """Export an Auto-Editor workflow file without rendering media.

    Args:
        source_media: Media file Auto-Editor analyzes.
        output_dir: Directory where the workflow file is written.
        output_stem: Base filename for the workflow export.
        mode: Auto-Editor workflow export mode.
        trim_settings: Silence trimming settings passed to Auto-Editor.
        process_callback: Optional progress/status callback.
        media_reference: Optional media path to embed in FCPXML outputs.

    Returns:
        Path to the exported workflow file.
    """

    normalized_mode = normalize_workflow_export_mode(mode)
    if not workflow_export_enabled(normalized_mode):
        raise ValueError("Auto-Editor workflow export mode must not be None.")

    source = Path(source_media).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{output_stem}.{_WORKFLOW_EXPORT_EXTENSIONS[normalized_mode]}"
    margin = f"{trim_settings.margin_seconds:g}s,{trim_settings.margin_seconds:g}s"
    emit_process_message(
        process_callback,
        f"Auto-Editor workflow export selected: {workflow_export_label(normalized_mode)}",
    )
    cmd = [
        *auto_editor_command(),
        str(source),
        "--no-open",
        "--margin",
        margin,
        "--edit",
        f"audio:threshold={trim_settings.threshold_db:g}dB",
        "--smooth",
        f"{trim_settings.mincut},{trim_settings.minclip}",
        "--export",
        normalized_mode,
        "-o",
        str(target),
        "--progress",
        "ascii",
    ]
    run_command(
        cmd,
        "auto-editor workflow export failed",
        process_callback=process_callback,
    )
    rewrite_fcpxml_media_references(target, media_reference or source)
    return str(target).replace("\\", "/")


def workflow_export_label(mode: str) -> str:
    """Return the display label for a workflow export mode."""

    normalized_mode = normalize_workflow_export_mode(mode)
    for label, value in AUTO_EDITOR_WORKFLOW_EXPORT_CHOICES:
        if value == normalized_mode:
            return label
    return "None"
