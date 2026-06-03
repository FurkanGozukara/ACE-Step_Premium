"""Inline latest-result preview wiring helpers."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.i18n import t

_GENERATED_AUDIO_OUTPUT_INDEX = 0
_GENERATED_FILES_OUTPUT_INDEX = 8
_STATUS_OUTPUT_INDEX = 10


def build_inline_result_outputs(generation_section: dict[str, Any]) -> list[Any]:
    """Return ordered outputs for the inline latest-result preview."""

    return [
        generation_section["inline_generated_audio"],
        generation_section["inline_remaining_audio"],
        generation_section["inline_generation_status"],
    ]


def clear_inline_result_preview() -> tuple[Any, Any, str]:
    """Clear the inline latest-result preview before a new generation starts."""

    return (
        gr.update(value=None, label=t("generation.inline_result_audio_label")),
        gr.update(value=None, visible=False),
        "",
    )


def prepare_inline_result_preview(task_type: str | None = None) -> tuple[Any, Any, str]:
    """Clear the inline preview and show Extract-mode foreground progress."""

    if str(task_type or "").strip().lower() == "extract":
        return (
            gr.update(value=None, label="Extracted Audio"),
            gr.update(value=None, visible=True),
            t("messages.extract_stem_processing"),
        )
    return clear_inline_result_preview()


def append_inline_result_preview(outputs: Any) -> tuple[Any, ...]:
    """Append inline latest-result audio and status to generation outputs."""

    output_tuple = tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
    return (*output_tuple, *inline_result_preview_from_generation_outputs(output_tuple))


def inline_result_preview_from_generation_outputs(outputs: Any) -> tuple[Any, Any, Any]:
    """Return inline audio/status updates from a streamed generation output tuple."""

    if not isinstance(outputs, (list, tuple)):
        return gr.skip(), gr.skip(), gr.skip()
    audio_update = _output_at(outputs, _GENERATED_AUDIO_OUTPUT_INDEX)
    remaining_update = _remaining_audio_update(
        _output_at(outputs, _GENERATED_FILES_OUTPUT_INDEX)
    )
    status_update = _status_update_from_output(_output_at(outputs, _STATUS_OUTPUT_INDEX))
    return audio_update, remaining_update, status_update


def sync_inline_result_preview(
    generated_audio: Any,
    generated_files: Any,
    status: Any,
) -> tuple[Any, Any, str]:
    """Mirror the first generated sample and status into the inline preview."""

    return generated_audio, _remaining_audio_update(generated_files), str(status or "")


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
    """Return a Gradio update for an ACE-Step Extract remaining-audio path."""

    path = _remaining_audio_path(paths)
    if not path:
        return gr.update(value=None, visible=False)
    return gr.update(value=path, visible=True)


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
