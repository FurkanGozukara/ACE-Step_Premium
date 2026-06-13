"""Metadata loading facade for generation handlers."""

from __future__ import annotations

import json
import os

import gradio as gr

from acestep.ui.gradio.i18n import t

from .metadata_examples import (
    load_random_example,
    load_random_simple_description,
    sample_example_smart,
)
from .metadata_restore_document import (
    generation_payload_from_document,
    resolve_metadata_path,
)
from .metadata_restore_values import metadata_values_from_payload, unchanged_metadata_values

__all__ = [
    "load_metadata",
    "load_metadata_with_status",
    "load_random_example",
    "load_random_simple_description",
    "sample_example_smart",
]


def load_metadata(file_obj, llm_handler=None):
    """Load generation parameters from a JSON file."""

    values, _status, loaded = _load_metadata_values(file_obj, "", llm_handler)
    return [*values, loaded]


def load_metadata_with_status(file_obj, path_value="", llm_handler=None):
    """Load generation metadata and append a human-readable status string."""

    values, status, loaded = _load_metadata_values(file_obj, path_value, llm_handler)
    return [*values, loaded, status]


def _load_metadata_values(file_obj, path_value, llm_handler=None):
    """Return ordered metadata values, status text, and load success."""

    filepath = resolve_metadata_path(file_obj, path_value)
    if not filepath:
        message = t("messages.no_file_selected")
        gr.Warning(message)
        return unchanged_metadata_values(), message, False

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        metadata = generation_payload_from_document(document)
        values = metadata_values_from_payload(metadata, llm_handler)
        gr.Info(t("messages.params_loaded", filename=os.path.basename(filepath)))
        return values, f"Loaded: {filepath}", True
    except json.JSONDecodeError as exc:
        message = t("messages.invalid_json", error=str(exc))
        gr.Warning(message)
        return unchanged_metadata_values(), message, False
    except Exception as exc:
        message = t("messages.load_error", error=str(exc))
        gr.Warning(message)
        return unchanged_metadata_values(), message, False
