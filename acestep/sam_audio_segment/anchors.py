"""Temporal anchor parsing for SAM-Audio span prompting."""

from __future__ import annotations

import json
from typing import Any

from .settings import SamAudioSettings

Anchor = tuple[str, float, float]


def anchors_for_settings(settings: SamAudioSettings) -> list[list[Anchor]] | None:
    """Return batch-wrapped anchors for a single input, or ``None``."""

    if settings.prompt_mode != "span" and not settings.use_span_anchor:
        return None
    anchors = _parse_anchor_json(settings.anchor_json)
    if anchors:
        return [anchors]
    start = max(0.0, float(settings.anchor_start))
    end = max(start, float(settings.anchor_end))
    if end <= start:
        return None
    return [[(settings.anchor_polarity, start, end)]]


def _parse_anchor_json(raw: str) -> list[Anchor]:
    """Parse a JSON anchor list such as ``[["+", 1.0, 2.0]]``."""

    text = raw.strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Anchor JSON must be a list.")
    anchors: list[Anchor] = []
    for item in parsed:
        anchors.append(_coerce_anchor(item))
    return anchors


def _coerce_anchor(item: Any) -> Anchor:
    """Validate and coerce one anchor item."""

    if not isinstance(item, (list, tuple)) or len(item) != 3:
        raise ValueError("Each anchor must be [polarity, start_seconds, end_seconds].")
    polarity = str(item[0]).strip()
    if polarity not in {"+", "-"}:
        raise ValueError("Anchor polarity must be '+' or '-'.")
    start = float(item[1])
    end = float(item[2])
    if start < 0 or end <= start:
        raise ValueError("Anchor times must satisfy 0 <= start < end.")
    return polarity, start, end
