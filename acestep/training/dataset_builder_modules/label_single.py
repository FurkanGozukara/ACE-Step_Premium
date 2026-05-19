"""Single-sample auto-labeling helpers for training datasets."""

from typing import Optional, Tuple

from loguru import logger

from .label_utils import get_audio_codes, parse_int
from .lyrics_quality import select_training_lyrics
from .models import AudioSample


_FORMAT_LYRICS_REPETITION_PENALTY = 1.18
_FORMAT_LYRICS_TEMPERATURE = 0.20
_FORMAT_LYRICS_TOP_P = 0.75
_TRANSCRIBE_TEMPERATURE = 0.1
_TRANSCRIBE_TOP_P = 0.3


def _clean_llm_lyrics(lyrics: object) -> str:
    """Return usable lyrics text from an LLM metadata value."""

    if not isinstance(lyrics, str):
        return ""
    lyrics = lyrics.strip()
    return "" if lyrics == "[Instrumental]" else lyrics


def _normalize_language_hint(language: object) -> str:
    """Return a valid LM language hint or an empty string for auto-detection."""

    if not isinstance(language, str):
        return ""
    language = language.strip()
    if not language or language.lower() in {"unknown", "instrumental", "auto"}:
        return ""
    return language


def _language_metadata(language: str) -> dict[str, str] | None:
    """Return constrained-decoding metadata for a selected lyrics language."""

    language = _normalize_language_hint(language)
    return {"language": language} if language else None


def _format_language_metadata(
    requested_language: str,
    sample: AudioSample,
    metadata: dict | None,
) -> dict[str, str] | None:
    """Return the best language constraint for formatting existing lyrics."""

    language = _normalize_language_hint(requested_language)
    if not language and metadata:
        language = _normalize_language_hint(
            metadata.get("language") or metadata.get("vocal_language")
        )
    if not language:
        language = _normalize_language_hint(sample.language)
    return {"language": language} if language else None


def _apply_audio_metadata(
    sample: AudioSample,
    metadata: dict,
    *,
    skip_metas: bool,
    has_csv_bpm: bool,
    has_csv_key: bool,
) -> None:
    """Apply audio-inferred metadata while preserving user/CSV overrides."""

    sample.caption = metadata.get("caption", "")
    sample.genre = metadata.get("genres", "")

    if not skip_metas:
        if not has_csv_bpm:
            sample.bpm = parse_int(metadata.get("bpm"))
        if not has_csv_key:
            sample.keyscale = metadata.get("keyscale", "")
        sample.timesignature = metadata.get("timesignature", "")

    sample.language = metadata.get(
        "language",
        metadata.get("vocal_language", sample.language or "unknown"),
    )


class LabelSingleMixin:
    """Label a single sample."""

    def label_sample(
        self,
        sample_idx: int,
        dit_handler,
        llm_handler,
        format_lyrics: bool = False,
        transcribe_lyrics: bool = False,
        lm_lyrics_language: str = "unknown",
        skip_metas: bool = False,
        progress_callback=None,
    ) -> Tuple[Optional[AudioSample], str]:
        """Label a single sample using the LLM."""
        if sample_idx < 0 or sample_idx >= len(self.samples):
            return None, f"❌ Invalid sample index: {sample_idx}"

        sample = self.samples[sample_idx]

        has_preloaded_lyrics = sample.has_raw_lyrics() and not sample.is_instrumental
        has_csv_bpm = sample.bpm is not None
        has_csv_key = bool(sample.keyscale)

        try:
            if progress_callback:
                progress_callback(f"Processing: {sample.filename}")

            audio_codes = get_audio_codes(sample.audio_path, dit_handler)

            if not audio_codes:
                return sample, f"❌ Failed to encode audio: {sample.filename}"

            if progress_callback:
                progress_callback(f"Generating metadata for: {sample.filename}")

            metadata = None
            metadata_status = ""
            needs_audio_metadata = transcribe_lyrics or not (
                format_lyrics and has_preloaded_lyrics
            )
            if needs_audio_metadata:
                metadata, metadata_status = llm_handler.understand_audio_from_codes(
                    audio_codes=audio_codes,
                    temperature=_TRANSCRIBE_TEMPERATURE if transcribe_lyrics else 0.7,
                    top_p=_TRANSCRIBE_TOP_P if transcribe_lyrics else None,
                    user_metadata=_language_metadata(lm_lyrics_language)
                    if transcribe_lyrics
                    else None,
                    use_constrained_decoding=True,
                )

                if not metadata:
                    return sample, f"❌ LLM labeling failed: {metadata_status}"

            if format_lyrics and has_preloaded_lyrics:
                from acestep.inference import format_sample

                result = format_sample(
                    llm_handler=llm_handler,
                    caption=metadata.get("caption", "") if metadata else "",
                    lyrics=sample.raw_lyrics,
                    user_metadata=_format_language_metadata(
                        lm_lyrics_language,
                        sample,
                        metadata,
                    ),
                    temperature=_FORMAT_LYRICS_TEMPERATURE,
                    top_p=_FORMAT_LYRICS_TOP_P,
                    repetition_penalty=_FORMAT_LYRICS_REPETITION_PENALTY,
                    use_constrained_decoding=True,
                )

                if not result.success:
                    return sample, f"❌ LLM format failed: {result.error}"

                if metadata is not None:
                    _apply_audio_metadata(
                        sample,
                        metadata,
                        skip_metas=skip_metas,
                        has_csv_bpm=has_csv_bpm,
                        has_csv_key=has_csv_key,
                    )
                else:
                    sample.caption = result.caption or ""
                    if not skip_metas:
                        if not has_csv_bpm:
                            sample.bpm = result.bpm
                        if not has_csv_key:
                            sample.keyscale = result.keyscale or ""
                        sample.timesignature = result.timesignature or ""
                    sample.language = result.language or "unknown"

                language_hint = _normalize_language_hint(lm_lyrics_language)
                if language_hint:
                    sample.language = language_hint
                lyrics_selection = select_training_lyrics(
                    sample.raw_lyrics,
                    _clean_llm_lyrics(result.lyrics),
                )
                sample.formatted_lyrics = lyrics_selection.formatted_lyrics
                sample.lyrics = lyrics_selection.lyrics
                sample.is_instrumental = False

                if metadata is not None:
                    status_suffix = "(lyrics from file; metadata inferred from audio)"
                    if lyrics_selection.rejection_reason:
                        status_suffix = (
                            "(lyrics from file; metadata inferred from audio; "
                            f"LM format rejected: {lyrics_selection.rejection_reason})"
                        )
                else:
                    status_suffix = (
                        "(lyrics formatted by LM)"
                        if sample.formatted_lyrics
                        else "(using raw lyrics; LM returned no lyrics)"
                    )
                    if lyrics_selection.rejection_reason:
                        status_suffix = (
                            "(using raw lyrics; "
                            f"LM format rejected: {lyrics_selection.rejection_reason})"
                        )

            else:
                _apply_audio_metadata(
                    sample,
                    metadata,
                    skip_metas=skip_metas,
                    has_csv_bpm=has_csv_bpm,
                    has_csv_key=has_csv_key,
                )

                llm_lyrics = _clean_llm_lyrics(metadata.get("lyrics", ""))

                if sample.is_instrumental and not transcribe_lyrics:
                    sample.lyrics = "[Instrumental]"
                    sample.language = "unknown"
                    sample.formatted_lyrics = ""
                    status_suffix = "(instrumental)"
                elif transcribe_lyrics:
                    if has_preloaded_lyrics:
                        sample.lyrics = sample.raw_lyrics
                        sample.formatted_lyrics = ""
                        language_hint = _normalize_language_hint(lm_lyrics_language)
                        if language_hint:
                            sample.language = language_hint
                        sample.is_instrumental = False
                        status_suffix = "(using raw lyrics; metadata inferred from audio)"
                    else:
                        lyrics_selection = select_training_lyrics(sample.raw_lyrics, llm_lyrics)
                        sample.formatted_lyrics = lyrics_selection.formatted_lyrics
                        if sample.formatted_lyrics:
                            sample.lyrics = sample.formatted_lyrics
                            language_hint = _normalize_language_hint(lm_lyrics_language)
                            if language_hint:
                                sample.language = language_hint
                            sample.is_instrumental = False
                            status_suffix = "(lyrics transcribed by LM)"
                        elif lyrics_selection.lyrics:
                            sample.lyrics = lyrics_selection.lyrics
                            sample.formatted_lyrics = ""
                            sample.is_instrumental = False
                            status_suffix = (
                                "(using cleaned raw lyrics; "
                                f"LM transcription rejected: {lyrics_selection.rejection_reason})"
                            )
                        else:
                            sample.lyrics = "[Instrumental]"
                            sample.language = "unknown"
                            sample.formatted_lyrics = ""
                            sample.is_instrumental = True
                            status_suffix = (
                                "(LM transcription rejected: "
                                f"{lyrics_selection.rejection_reason})"
                                if lyrics_selection.rejection_reason
                                else "(no lyrics transcribed)"
                            )
                elif has_preloaded_lyrics:
                    sample.lyrics = sample.raw_lyrics
                    sample.formatted_lyrics = ""
                    status_suffix = "(using raw lyrics)"
                else:
                    if llm_lyrics:
                        sample.lyrics = llm_lyrics
                        sample.formatted_lyrics = llm_lyrics
                        sample.is_instrumental = False
                        status_suffix = ""
                    else:
                        sample.lyrics = "[Instrumental]"
                        sample.language = "unknown"
                        sample.formatted_lyrics = ""
                        sample.is_instrumental = True
                        status_suffix = "(instrumental)"

            sample.labeled = True
            self.samples[sample_idx] = sample

            status_msg = f"✅ Labeled: {sample.filename}"
            if skip_metas:
                status_msg += " (skip metas)"
            if status_suffix:
                status_msg += f" {status_suffix}"

            return sample, status_msg

        except Exception as e:
            logger.exception(f"Error labeling sample {sample.filename}")
            return sample, f"❌ Error: {str(e)}"
