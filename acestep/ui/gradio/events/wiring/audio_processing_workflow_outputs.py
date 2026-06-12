"""Workflow export status helpers for Audio Processing UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from acestep.audio_processing.auto_editor_workflow import workflow_export_label
from acestep.ui.gradio.events.wiring.audio_processing_source_paths import (
    workflow_reference_note,
)


def workflow_export_markdown(
    input_path: str,
    workflow_path: str,
    mode: str,
    media_reference: str | None = None,
    local_path_value: Any = None,
) -> str:
    """Return UI status for an Auto-Editor workflow-only export."""

    reference = media_reference or input_path
    lines = [
        "### Auto-Editor Workflow Export",
        "**Export complete.**",
        f"- Source analyzed: `{Path(input_path).name}`",
        f"- Workflow: `{workflow_export_label(mode)}`",
        f"- Exported file: `{workflow_path}`",
        "- Processed audio/video: `None`",
    ]
    note = workflow_reference_note(input_path, reference, local_path_value)
    if note:
        lines.append(f"- {note}")
    return "\n".join(lines)
