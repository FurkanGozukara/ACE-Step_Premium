"""Batch-segment prompt parsing for SAM-Audio."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .seed import resolve_runtime_seed
from .settings import SamAudioSettings


@dataclass(frozen=True)
class BatchSegmentPrompt:
    """One normalized SAM-Audio batch-segment prompt."""

    text: str
    suffix: str


def batch_segment_prompts(settings: SamAudioSettings) -> list[BatchSegmentPrompt]:
    """Return normalized prompts when Batch Segment behavior is active."""

    if not is_batch_segment_active(settings):
        return []
    if settings.custom_prompt.strip():
        return parse_batch_segment_prompts(settings.custom_prompt)
    return parse_batch_segment_prompts(settings.prompt_presets)


def is_batch_segment_active(settings: SamAudioSettings) -> bool:
    """Return whether this request should run one pass per prompt."""

    if settings.batch_segment:
        return True
    if settings.custom_prompt.strip():
        return False
    return len(settings.prompt_presets) > 1


def parse_batch_segment_prompts(raw_value: Any) -> list[BatchSegmentPrompt]:
    """Parse semicolon-separated prompt text into model prompts and file suffixes."""

    prompts: list[BatchSegmentPrompt] = []
    suffix_counts: dict[str, int] = {}
    for raw_part in _raw_prompt_parts(raw_value):
        prompt = normalize_batch_segment_prompt(raw_part)
        if not prompt:
            continue
        base_suffix = batch_segment_suffix(prompt)
        suffix_counts[base_suffix] = suffix_counts.get(base_suffix, 0) + 1
        suffix = (
            base_suffix
            if suffix_counts[base_suffix] == 1
            else f"{base_suffix}_{suffix_counts[base_suffix]}"
        )
        prompts.append(BatchSegmentPrompt(text=prompt, suffix=suffix))
    if not prompts:
        raise ValueError(
            "Batch Segment is enabled. Enter semicolon-separated prompts in Custom "
            "Prompt or select one or more Quick Prompt values."
        )
    return prompts


def _raw_prompt_parts(raw_value: Any) -> Iterable[Any]:
    """Yield prompt entries from semicolon text or multiselect dropdown values."""

    if isinstance(raw_value, (list, tuple, set)):
        for item in raw_value:
            if isinstance(item, str):
                yield from item.split(";")
            else:
                yield item
        return
    yield from str(raw_value or "").split(";")


def normalize_batch_segment_prompt(value: Any) -> str:
    """Return prompt text normalized for SAM-Audio text conditioning."""

    normalized = str(value or "").strip().lower()
    normalized = normalized.strip(" ;,")
    return re.sub(r"\s+", " ", normalized)


def batch_segment_suffix(prompt: str) -> str:
    """Return a filesystem-safe suffix for one normalized prompt."""

    suffix = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    suffix = re.sub(r"_+", "_", suffix)
    return suffix[:80].strip("_") or "segment"


def settings_for_batch_segment_prompt(
    settings: SamAudioSettings,
    prompt: BatchSegmentPrompt,
) -> SamAudioSettings:
    """Return settings for one concrete batch-segment prompt."""

    return resolve_runtime_seed(
        replace(
            settings,
            custom_prompt=prompt.text,
            batch_segment=False,
        )
    )
