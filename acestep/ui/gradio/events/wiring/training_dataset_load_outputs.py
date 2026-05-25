"""Shared output contracts for training dataset JSON load handlers."""

from __future__ import annotations

from typing import Any, Mapping

import gradio as gr


DATASET_LOAD_SHARED_OUTPUT_KEYS = (
    "audio_files_table",
    "sample_selector",
    "dataset_builder_state",
    "preview_audio",
    "preview_filename",
    "edit_caption",
    "edit_genre",
    "prompt_override",
    "edit_lyrics",
    "edit_bpm",
    "edit_keyscale",
    "edit_timesig",
    "edit_duration",
    "edit_language",
    "edit_instrumental",
    "raw_lyrics_display",
    "has_raw_lyrics_state",
    "dataset_name",
    "custom_tag",
    "tag_position",
    "all_instrumental",
    "genre_ratio",
    "use_only_custom_trigger",
)


def build_dataset_load_outputs(
    training_section: Mapping[str, Any],
    status_key: str,
) -> list[Any]:
    """Return ordered Gradio outputs for a dataset JSON load event."""

    return [training_section[status_key]] + [
        training_section[key] for key in DATASET_LOAD_SHARED_OUTPUT_KEYS
    ]


def no_dataset_load_outputs(status: str) -> tuple[Any, ...]:
    """Return no-op load outputs with a status message."""

    return (status, *[gr.update() for _ in DATASET_LOAD_SHARED_OUTPUT_KEYS])
