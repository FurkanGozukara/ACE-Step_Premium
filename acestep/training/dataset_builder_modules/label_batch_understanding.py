"""Batched LM understanding helpers for dataset auto-labeling."""

from __future__ import annotations

from typing import Any

from .label_single import _language_metadata


_SUCCESS = "\u2705"
_TRANSCRIBE_TEMPERATURE = 0.1
_TRANSCRIBE_TOP_P = 0.3


def understand_audio_codes_batch(
    llm_handler: object,
    audio_codes_batch: list[str],
    *,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
) -> list[tuple[dict[str, Any], str]]:
    """Return LM metadata for one batch of audio-code strings."""

    if len(audio_codes_batch) == 1:
        return [
            _understand_one(
                llm_handler,
                audio_codes_batch[0],
                transcribe_lyrics=transcribe_lyrics,
                lm_lyrics_language=lm_lyrics_language,
            )
        ]

    if not _supports_prompt_batching(llm_handler):
        return [
            _understand_one(
                llm_handler,
                audio_codes,
                transcribe_lyrics=transcribe_lyrics,
                lm_lyrics_language=lm_lyrics_language,
            )
            for audio_codes in audio_codes_batch
        ]

    prompts = [
        llm_handler.build_formatted_prompt_for_understanding(audio_codes)
        for audio_codes in audio_codes_batch
    ]
    output_texts, status = llm_handler.generate_from_formatted_prompt(
        formatted_prompt=prompts,
        cfg={
            "temperature": _TRANSCRIBE_TEMPERATURE if transcribe_lyrics else 0.7,
            "top_p": _TRANSCRIBE_TOP_P if transcribe_lyrics else None,
            "generation_phase": "understand",
            "caption": "",
            "lyrics": "",
        },
        use_constrained_decoding=False,
        constrained_decoding_debug=False,
        stop_at_reasoning=False,
    )
    output_list = _as_output_list(output_texts, len(audio_codes_batch))
    results = [
        _metadata_from_output(llm_handler, output, status)
        for output in output_list
    ]
    for index, (metadata, _status) in enumerate(results):
        if not metadata:
            results[index] = _understand_one(
                llm_handler,
                audio_codes_batch[index],
                transcribe_lyrics=transcribe_lyrics,
                lm_lyrics_language=lm_lyrics_language,
            )
    return results


def _understand_one(
    llm_handler: object,
    audio_codes: str,
    *,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
) -> tuple[dict[str, Any], str]:
    """Run the existing single-item understand path."""

    return llm_handler.understand_audio_from_codes(
        audio_codes=audio_codes,
        temperature=_TRANSCRIBE_TEMPERATURE if transcribe_lyrics else 0.7,
        top_p=_TRANSCRIBE_TOP_P if transcribe_lyrics else None,
        user_metadata=_language_metadata(lm_lyrics_language) if transcribe_lyrics else None,
        use_constrained_decoding=True,
    )


def _supports_prompt_batching(llm_handler: object) -> bool:
    """Return whether the LM handler exposes the prompt-level batch API."""

    return all(
        hasattr(llm_handler, name)
        for name in (
            "build_formatted_prompt_for_understanding",
            "generate_from_formatted_prompt",
            "parse_lm_output",
        )
    )


def _as_output_list(output_texts: object, expected_count: int) -> list[str]:
    """Normalize LM batch output to the expected list length."""

    if isinstance(output_texts, list):
        outputs = [str(output or "") for output in output_texts]
    elif isinstance(output_texts, str):
        outputs = [output_texts]
    else:
        outputs = []
    if len(outputs) < expected_count:
        outputs.extend([""] * (expected_count - len(outputs)))
    return outputs[:expected_count]


def _metadata_from_output(
    llm_handler: object,
    output_text: str,
    status: str,
) -> tuple[dict[str, Any], str]:
    """Parse one LM batch output into metadata and status."""

    if not output_text:
        return {}, status
    metadata, _audio_codes = llm_handler.parse_lm_output(output_text)
    extract_lyrics = getattr(llm_handler, "_extract_lyrics_from_output", None)
    if callable(extract_lyrics):
        lyrics = extract_lyrics(output_text)
        if lyrics:
            metadata["lyrics"] = lyrics
    if not metadata:
        return {}, status
    return metadata, f"{_SUCCESS} Understanding completed successfully"
