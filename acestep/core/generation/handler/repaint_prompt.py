"""Repaint prompt helpers for lyric-targeted masked edits."""

import re
from typing import Any


REPAINT_LYRICS_INFO_TEXT = (
    "Lyrics and section structure for the target result. In Repaint, provide "
    "the words or vocal direction for the selected replacement range."
)


def has_repaint_lyrics(lyrics: str | None) -> bool:
    """Return whether repaint received usable target lyrics."""
    lyric_text = str(lyrics or "").strip()
    if not lyric_text or "[instrumental]" in lyric_text.lower():
        return False
    return not _is_repaint_helper_text(lyric_text)


def normalize_repaint_lyrics(task_type: str, lyrics: str | None) -> str:
    """Return sanitized lyrics for repaint requests.

    Args:
        task_type: Current generation task type.
        lyrics: Raw lyric text from the UI or API.

    Returns:
        The original lyric text, except known helper copy is treated as empty
        for Repaint so it cannot be conditioned as target words.
    """
    lyric_text = str(lyrics or "")
    if task_type == "repaint" and _is_repaint_helper_text(lyric_text):
        return ""
    return lyric_text


def resolve_repaint_span_duration(
    task_type: str,
    repainting_start: float | None,
    repainting_end: float | None,
    lyrics: str | None,
) -> float | None:
    """Return selected repaint duration when lyric conditioning should be local."""
    if task_type != "repaint" or not has_repaint_lyrics(lyrics):
        return None
    try:
        start = float(repainting_start or 0.0)
        end = float(repainting_end) if repainting_end is not None else None
    except (TypeError, ValueError):
        return None
    if end is None or end <= start:
        return None
    return end - start


def apply_repaint_span_duration_to_metas(
    task_type: str,
    metas_batch: list[Any],
    repainting_start: float | None,
    repainting_end: float | None,
    lyrics: str | None,
) -> list[Any]:
    """Use the selected mask duration in Repaint metadata when lyrics are local.

    Args:
        task_type: Current generation task type.
        metas_batch: Batch metadata dictionaries or strings.
        repainting_start: Selected repaint start time in seconds.
        repainting_end: Selected repaint end time in seconds.
        lyrics: Target lyrics intended for the selected range.

    Returns:
        Metadata with only the duration field changed for lyric-bearing Repaint.
    """
    span_duration = resolve_repaint_span_duration(
        task_type,
        repainting_start,
        repainting_end,
        lyrics,
    )
    if span_duration is None:
        return metas_batch

    duration_text = f"{max(1, int(round(span_duration)))} seconds"
    updated_metas = []
    for meta in metas_batch:
        if isinstance(meta, dict):
            updated = meta.copy()
            updated["duration"] = duration_text
            updated_metas.append(updated)
        else:
            updated_metas.append(meta)
    return updated_metas


def strengthen_repaint_caption(
    task_type: str,
    caption: str,
    lyrics: str,
    vocal_language: str,
    span_duration: float | None = None,
) -> str:
    """Add an explicit target-lyric hint to repaint caption conditioning.

    Args:
        task_type: Current generation task type.
        caption: User style/caption text.
        lyrics: Target lyrics intended for the repainted region.
        vocal_language: Requested vocal language code.
        span_duration: Optional selected repaint duration in seconds.

    Returns:
        Caption text, augmented only for lyric-bearing repaint requests.
    """
    if task_type != "repaint" or not has_repaint_lyrics(lyrics):
        return caption

    lyric_line = _normalize_inline_text(lyrics)
    if not lyric_line:
        return caption

    language_hint = _language_hint(vocal_language)
    span_hint = _span_duration_hint(span_duration)
    target_hint = (
        f" Repaint the selected{span_hint} mask with a clear {language_hint} "
        f"vocal singing exactly: "
        f'"{lyric_line}".'
    )
    return f"{str(caption or '').strip()}{target_hint}".strip()


def resolve_repaint_vocal_language(
    task_type: str,
    vocal_language: str,
    lyrics: str,
) -> str:
    """Default lyric-bearing repaint to English when language is automatic."""
    language = str(vocal_language or "").strip()
    if task_type == "repaint" and has_repaint_lyrics(lyrics):
        if not language or language.lower() == "unknown":
            return "en"
    return vocal_language


def resolve_repaint_chunk_mask_mode(
    task_type: str,
    chunk_mask_mode: str,
    lyrics: str,
) -> str:
    """Use explicit mask conditioning for lyric-bearing repaint requests."""
    if task_type == "repaint" and has_repaint_lyrics(lyrics):
        return "explicit"
    return chunk_mask_mode


def _normalize_inline_text(text: str) -> str:
    """Collapse lyric text to a short inline phrase for caption hints."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_repaint_helper_text(text: str) -> bool:
    """Return whether text is the UI Repaint lyric guidance sentence."""
    return _normalize_inline_text(text).lower() == REPAINT_LYRICS_INFO_TEXT.lower()


def _span_duration_hint(span_duration: float | None) -> str:
    """Return a short selected-mask duration hint for caption conditioning."""
    if span_duration is None:
        return ""
    seconds = max(1, int(round(span_duration)))
    return f" {seconds}-second"


def _language_hint(vocal_language: str) -> str:
    """Return a readable language hint for short repaint captions."""
    language = str(vocal_language or "").strip().lower()
    if language == "en":
        return "English"
    if language and language != "unknown":
        return language
    return "English"
