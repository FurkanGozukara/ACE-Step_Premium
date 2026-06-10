"""Inline latest-result preview wiring helpers."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.events.results.result_output_contract import (
    ALL_AUDIO_PATHS_INDEX,
    STATUS_INDEX,
    extract_latest_edit_area_paths,
    extract_source_audio_path,
)

_GENERATED_AUDIO_OUTPUT_INDEX = 0
_GENERATED_FILES_OUTPUT_INDEX = ALL_AUDIO_PATHS_INDEX
_STATUS_OUTPUT_INDEX = STATUS_INDEX


def build_inline_result_outputs(generation_section: dict[str, Any]) -> list[Any]:
    """Return ordered outputs for the inline latest-result preview."""

    return [
        generation_section["inline_generated_audio"],
        generation_section["inline_remaining_audio"],
        generation_section["inline_repainted_area_audio"],
        generation_section["inline_repainted_area_original_audio"],
        generation_section["inline_generation_status"],
    ]


def clear_inline_result_preview() -> tuple[Any, Any, Any, Any, str]:
    """Clear the inline latest-result preview before a new generation starts."""

    return (
        gr.update(value=None, label=t("generation.inline_result_audio_label")),
        gr.update(value=None, visible=False),
        _hidden_repainted_area_update(),
        _hidden_repainted_area_original_update(),
        "",
    )


def prepare_inline_result_preview(
    task_type: str | None = None,
) -> tuple[Any, Any, Any, Any, str]:
    """Clear the inline preview and show Extract-mode foreground progress."""

    if str(task_type or "").strip().lower() == "extract":
        return (
            gr.update(value=None, label="Extracted Audio"),
            gr.update(value=None, visible=True),
            _hidden_repainted_area_update(),
            _hidden_repainted_area_original_update(),
            t("messages.extract_stem_processing"),
        )
    return clear_inline_result_preview()


def append_inline_result_preview(outputs: Any) -> tuple[Any, ...]:
    """Append inline latest-result audio and status to generation outputs."""

    output_tuple = tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
    return (*output_tuple, *inline_result_preview_from_generation_outputs(output_tuple))


def inline_result_preview_from_generation_outputs(outputs: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Return inline audio/status updates from a streamed generation output tuple."""

    if not isinstance(outputs, (list, tuple)):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    audio_update = _output_at(outputs, _GENERATED_AUDIO_OUTPUT_INDEX)
    generated_files = _output_at(outputs, _GENERATED_FILES_OUTPUT_INDEX)
    remaining_update = _remaining_audio_update(
        generated_files
    )
    edit_area_update, edit_area_original_update = _edit_area_updates(generated_files)
    status_update = _status_update_from_output(_output_at(outputs, _STATUS_OUTPUT_INDEX))
    return audio_update, remaining_update, edit_area_update, edit_area_original_update, status_update


def sync_inline_result_preview(
    generated_audio: Any,
    generated_files: Any,
    status: Any,
) -> tuple[Any, Any, Any, Any, str]:
    """Mirror the first generated sample and status into the inline preview."""

    edit_area_update, edit_area_original_update = _edit_area_updates(generated_files)
    return (
        generated_audio,
        _remaining_audio_update(generated_files),
        edit_area_update,
        edit_area_original_update,
        str(status or ""),
    )


def _output_at(outputs: list[Any] | tuple[Any, ...], index: int) -> Any:
    """Return a streamed output value or a no-op update when the index is absent."""

    if len(outputs) <= index:
        return gr.skip()
    return outputs[index]


def _status_update_from_output(status: Any) -> Any:
    """Normalize status output for the inline textbox while preserving no-op updates."""

    if _is_noop_update(status):
        return gr.skip()
    if isinstance(status, dict) and "value" in status:
        return str(status.get("value") or "")
    return str(status or "")


def _remaining_audio_update(paths: Any) -> Any:
    """Return an update for original source audio, falling back to remaining audio."""

    path = extract_source_audio_path(paths) or _remaining_audio_path(paths)
    if not path:
        return gr.update(value=None, visible=False)
    return gr.update(value=path, label="Original Input", visible=True)


def _edit_area_updates(paths: Any) -> tuple[Any, Any]:
    """Return updates for generated/original latest edited-area players."""

    generated_path, original_path = extract_latest_edit_area_paths(paths)
    if not generated_path or not original_path:
        return _hidden_repainted_area_update(), _hidden_repainted_area_original_update()
    return (
        gr.update(value=generated_path, label="Latest Repainted Area", visible=True),
        gr.update(
            value=original_path,
            label="Latest Repainted Area Original",
            visible=True,
        ),
    )


def _remaining_audio_path(paths: Any) -> str:
    """Return the first saved remaining-audio path from a generation file list."""

    if not isinstance(paths, list):
        return ""
    for candidate in paths:
        text = str(candidate or "").replace("\\", "/")
        name = text.rsplit("/", 1)[-1].lower()
        if "_remaining." in name and name.rsplit(".", 1)[-1] in {"wav", "mp3", "flac"}:
            return text
    return ""


def _is_noop_update(value: Any) -> bool:
    """Return whether a Gradio update intentionally leaves a component unchanged."""

    return isinstance(value, dict) and value == {"__type__": "update"}


def _hidden_repainted_area_update() -> Any:
    """Return a hidden update for the generated edited-area player."""

    return gr.update(value=None, label="Latest Repainted Area", visible=False)


def _hidden_repainted_area_original_update() -> Any:
    """Return a hidden update for the original edited-area player."""

    return gr.update(value=None, label="Latest Repainted Area Original", visible=False)
