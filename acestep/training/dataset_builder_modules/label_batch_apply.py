"""Apply batched auto-label metadata to dataset samples."""

from __future__ import annotations

from typing import Any

from .label_single import _apply_audio_metadata, _clean_llm_lyrics, _normalize_language_hint
from .lyrics_quality import select_training_lyrics
from .models import AudioSample


_SUCCESS = "\u2705"
_FAILURE = "\u274c"


def requires_single_label_path(
    sample: AudioSample,
    *,
    format_lyrics: bool,
) -> bool:
    """Return whether a sample needs the legacy per-item formatting path."""

    return bool(format_lyrics and sample.has_raw_lyrics() and not sample.is_instrumental)


def apply_understood_metadata(
    sample: AudioSample,
    metadata: dict[str, Any],
    *,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
    skip_metas: bool,
) -> tuple[AudioSample, str]:
    """Apply one batched LM metadata result to a sample."""

    if not metadata:
        return sample, f"{_FAILURE} LLM labeling failed: empty metadata"

    has_preloaded_lyrics = sample.has_raw_lyrics() and not sample.is_instrumental
    has_csv_bpm = sample.bpm is not None
    has_csv_key = bool(sample.keyscale)

    _apply_audio_metadata(
        sample,
        metadata,
        skip_metas=skip_metas,
        has_csv_bpm=has_csv_bpm,
        has_csv_key=has_csv_key,
    )
    status_suffix = _apply_lyrics(
        sample,
        _clean_llm_lyrics(metadata.get("lyrics", "")),
        transcribe_lyrics=transcribe_lyrics,
        lm_lyrics_language=lm_lyrics_language,
        has_preloaded_lyrics=has_preloaded_lyrics,
    )

    sample.labeled = True
    status = f"{_SUCCESS} Labeled: {sample.filename}"
    if skip_metas:
        status += " (skip metas)"
    if status_suffix:
        status += f" {status_suffix}"
    return sample, status


def _apply_lyrics(
    sample: AudioSample,
    llm_lyrics: str,
    *,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
    has_preloaded_lyrics: bool,
) -> str:
    """Apply lyrics according to the same rules as single-sample labeling."""

    if sample.is_instrumental and not transcribe_lyrics:
        sample.lyrics = "[Instrumental]"
        sample.language = "unknown"
        sample.formatted_lyrics = ""
        return "(instrumental)"
    if transcribe_lyrics:
        return _apply_transcribed_lyrics(
            sample,
            llm_lyrics,
            lm_lyrics_language=lm_lyrics_language,
            has_preloaded_lyrics=has_preloaded_lyrics,
        )
    if has_preloaded_lyrics:
        sample.lyrics = sample.raw_lyrics
        sample.formatted_lyrics = ""
        return "(using raw lyrics)"
    if llm_lyrics:
        sample.lyrics = llm_lyrics
        sample.formatted_lyrics = llm_lyrics
        sample.is_instrumental = False
        return ""
    sample.lyrics = "[Instrumental]"
    sample.language = "unknown"
    sample.formatted_lyrics = ""
    sample.is_instrumental = True
    return "(instrumental)"


def _apply_transcribed_lyrics(
    sample: AudioSample,
    llm_lyrics: str,
    *,
    lm_lyrics_language: str,
    has_preloaded_lyrics: bool,
) -> str:
    """Apply LM-transcribed lyrics or keep trusted raw lyrics."""

    if has_preloaded_lyrics:
        sample.lyrics = sample.raw_lyrics
        sample.formatted_lyrics = ""
        language_hint = _normalize_language_hint(lm_lyrics_language)
        if language_hint:
            sample.language = language_hint
        sample.is_instrumental = False
        return "(using raw lyrics; metadata inferred from audio)"

    lyrics_selection = select_training_lyrics(sample.raw_lyrics, llm_lyrics)
    sample.formatted_lyrics = lyrics_selection.formatted_lyrics
    if sample.formatted_lyrics:
        sample.lyrics = sample.formatted_lyrics
        language_hint = _normalize_language_hint(lm_lyrics_language)
        if language_hint:
            sample.language = language_hint
        sample.is_instrumental = False
        return "(lyrics transcribed by LM)"
    if lyrics_selection.lyrics:
        sample.lyrics = lyrics_selection.lyrics
        sample.formatted_lyrics = ""
        sample.is_instrumental = False
        return _rejected_transcription_status(lyrics_selection.rejection_reason)
    sample.lyrics = "[Instrumental]"
    sample.language = "unknown"
    sample.formatted_lyrics = ""
    sample.is_instrumental = True
    if lyrics_selection.rejection_reason:
        return f"(LM transcription rejected: {lyrics_selection.rejection_reason})"
    return "(no lyrics transcribed)"


def _rejected_transcription_status(reason: str | None) -> str:
    """Return status text for a rejected LM transcription with fallback lyrics."""

    return f"(using cleaned raw lyrics; LM transcription rejected: {reason})"
