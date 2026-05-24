"""Single-sample fallback helper for batched auto-label runs."""

from __future__ import annotations

from typing import Callable

from .label_batch_persistence import persist_successful_label
from .label_progress import replay_progress_after_llm_load
from .models import AudioSample


def label_single_fallback(
    builder: object,
    sample_idx: int,
    dit_handler: object,
    llm_handler: object,
    format_lyrics: bool,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
    skip_metas: bool,
    progress_callback: Callable[[str], None] | None,
    start_msg: str,
    persist_labels: bool,
    label_output_dir: str | None,
    label_source_root: str | None,
    sample_labeled_callback: Callable[[int, AudioSample, str], None] | None,
) -> tuple[str, bool]:
    """Run the existing per-item label path and persist a success."""

    with replay_progress_after_llm_load(llm_handler, progress_callback, start_msg):
        _sample, status = builder.label_sample(
            sample_idx,
            dit_handler,
            llm_handler,
            format_lyrics,
            transcribe_lyrics,
            lm_lyrics_language,
            skip_metas,
            progress_callback,
        )
    sidecar_failed = persist_successful_label(
        builder.samples[sample_idx],
        sample_idx,
        status,
        persist_labels,
        label_output_dir,
        label_source_root,
        sample_labeled_callback,
    )
    return status, sidecar_failed
