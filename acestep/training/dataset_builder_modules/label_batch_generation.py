"""Progress-wrapped LM metadata generation for batched auto-labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .label_batch_understanding import understand_audio_codes_batch
from .label_progress import progress_heartbeat, replay_progress_after_llm_load


METADATA_BATCH_HEARTBEAT_SECONDS = 5.0


@dataclass
class _BatchEntry:
    """Audio-code payload for one sample inside a batched LM request."""

    sample_idx: int
    filename: str
    audio_codes: str


def generate_metadata_batch(
    llm_handler: object,
    audio_codes_batch: list[str],
    *,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
    progress_callback: Callable[[str], None] | None,
) -> list[tuple[dict[str, Any], str]]:
    """Generate one batch of LM metadata with visible elapsed progress."""

    batch_msg = f"Generating metadata batch ({len(audio_codes_batch)} files)..."
    if progress_callback:
        progress_callback(batch_msg)
    with (
        replay_progress_after_llm_load(llm_handler, progress_callback, batch_msg),
        progress_heartbeat(
            progress_callback,
            batch_msg,
            interval_seconds=METADATA_BATCH_HEARTBEAT_SECONDS,
        ),
    ):
        return understand_audio_codes_batch(
            llm_handler,
            audio_codes_batch,
            transcribe_lyrics=transcribe_lyrics,
            lm_lyrics_language=lm_lyrics_language,
        )
