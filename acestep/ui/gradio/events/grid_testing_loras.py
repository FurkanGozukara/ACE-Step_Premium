"""LoRA selection helpers for Grid Testing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acestep.core.generation.handler.lora.folder_scan import (
    resolve_loadable_lora_adapter_path,
)


BASE_MODEL_LABEL = "None (base model)"
BASE_MODEL_PREFIX = "base-model"


@dataclass(frozen=True)
class GridLoraJob:
    """One Grid Testing LoRA or base-model generation job."""

    label: str
    path: str
    prefix: str


def resolve_grid_lora_jobs(selected_loras: Any) -> list[GridLoraJob]:
    """Return ordered, de-duplicated LoRA jobs for a grid run.

    Args:
        selected_loras: Raw Gradio multiselect value.

    Returns:
        A non-empty list of grid jobs. Empty input defaults to the base model.

    Raises:
        ValueError: If a non-empty selected LoRA path cannot be resolved.
    """

    raw_values = _normalize_selected_values(selected_loras) or [""]
    jobs: list[GridLoraJob] = []
    seen_paths: set[str] = set()
    used_prefixes: set[str] = set()

    for raw_value in raw_values:
        requested = str(raw_value or "").strip()
        if not requested:
            path = ""
            label = BASE_MODEL_LABEL
            prefix = BASE_MODEL_PREFIX
        else:
            path = resolve_loadable_lora_adapter_path(requested)
            if not path:
                raise ValueError(f"Invalid LoRA selection: {requested}")
            label = Path(path).name
            prefix = _prefix_from_lora_path(path)

        dedupe_key = path or "__base_model__"
        if dedupe_key in seen_paths:
            continue
        seen_paths.add(dedupe_key)

        unique_prefix = _unique_prefix(prefix, used_prefixes)
        used_prefixes.add(unique_prefix)
        jobs.append(GridLoraJob(label=label, path=path, prefix=unique_prefix))

    return jobs or [GridLoraJob(label=BASE_MODEL_LABEL, path="", prefix=BASE_MODEL_PREFIX)]


def filter_grid_lora_choices(
    choices: list[tuple[str, str]],
    filter_text: Any = "",
    selected_loras: Any = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Filter LoRA dropdown choices while preserving selected values.

    Args:
        choices: All available LoRA dropdown choices.
        filter_text: Case-insensitive text matched against labels and paths.
        selected_loras: Current multiselect value.

    Returns:
        ``(visible_choices, selected_values)`` for a Gradio dropdown update.
    """

    valid_choices = [(str(label), str(value or "")) for label, value in choices]
    choice_by_value = {value: (label, value) for label, value in valid_choices}
    selected = [
        value
        for value in _normalize_selected_values(selected_loras)
        if value in choice_by_value
    ]
    if not selected and "" in choice_by_value:
        selected = [""]

    needle = str(filter_text or "").strip().lower()
    if not needle:
        return valid_choices, selected

    filtered = [
        choice
        for choice in valid_choices
        if _choice_matches_filter(choice, needle)
    ]
    return _prepend_selected_choices(filtered, selected, choice_by_value), selected


def _normalize_selected_values(selected_loras: Any) -> list[str]:
    """Return selected LoRA values as a flat string list."""

    if selected_loras is None:
        return []
    if isinstance(selected_loras, str):
        return [selected_loras]
    if isinstance(selected_loras, (list, tuple, set)):
        return [str(value or "").strip() for value in selected_loras]
    return [str(selected_loras).strip()]


def _choice_matches_filter(choice: tuple[str, str], needle: str) -> bool:
    """Return whether a LoRA choice matches the filter text."""

    label, value = choice
    haystack = f"{label}\n{value}".lower()
    return needle in haystack


def _prepend_selected_choices(
    filtered: list[tuple[str, str]],
    selected: list[str],
    choice_by_value: dict[str, tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return filtered choices with hidden selections kept available."""

    seen = {value for _label, value in filtered}
    selected_choices = [
        choice_by_value[value]
        for value in selected
        if value in choice_by_value and value not in seen
    ]
    return [*selected_choices, *filtered]


def _prefix_from_lora_path(path: str) -> str:
    """Return a filesystem-safe filename prefix for a LoRA path."""

    candidate = Path(path).stem if Path(path).is_file() else Path(path).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-_")
    return cleaned or "lora"


def _unique_prefix(prefix: str, used_prefixes: set[str]) -> str:
    """Return a prefix that will not collide with previous grid jobs."""

    if prefix not in used_prefixes:
        return prefix
    suffix = 2
    while f"{prefix}-{suffix}" in used_prefixes:
        suffix += 1
    return f"{prefix}-{suffix}"
