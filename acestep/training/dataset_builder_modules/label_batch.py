"""Batched auto-label orchestration for dataset samples."""

from __future__ import annotations

from typing import Any, Callable

from .label_batch_apply import apply_understood_metadata, requires_single_label_path
from .label_batch_generation import _BatchEntry, generate_metadata_batch
from .label_batch_persistence import (
    cancelled_batch_status,
    common_audio_source_root,
    final_batch_status,
    finish_label_counts,
    persist_successful_label,
)
from .label_batch_progress import complete_batch_item, start_batch_item
from .label_batch_single import label_single_fallback
from .label_progress import LabelProgressTracker
from .label_utils import get_audio_codes
from .models import AudioSample


def label_samples_in_batches(
    builder: Any,
    *,
    dit_handler: object,
    llm_handler: object,
    samples_to_label: list[tuple[int, AudioSample]],
    batch_size: int,
    format_lyrics: bool,
    transcribe_lyrics: bool,
    lm_lyrics_language: str,
    skip_metas: bool,
    only_unlabeled: bool,
    progress_callback: Callable[[str], None] | None,
    sample_labeled_callback: Callable[[int, AudioSample, str], None] | None,
    persist_labels: bool,
    label_output_dir: str | None,
    label_source_root: str | None,
    initial_labeled_count: int,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[list[AudioSample], str]:
    """Label samples using grouped LM metadata requests when possible."""

    success_count = 0
    fail_count = 0
    sidecar_fail_count = 0
    processed_count = 0
    total_to_label = len(samples_to_label)
    skipped_count = len(builder.samples) - total_to_label if only_unlabeled else 0
    progress_tracker = LabelProgressTracker(len(builder.samples))
    resolved_source_root = (
        label_source_root
        or getattr(builder, "_current_dir", None)
        or common_audio_source_root(builder.samples)
    )

    for chunk_start in range(0, total_to_label, batch_size):
        if cancel_callback and cancel_callback():
            return builder.samples, cancelled_batch_status(
                success_count, total_to_label, total_to_label - processed_count
            )
        entries: list[_BatchEntry] = []
        chunk = samples_to_label[chunk_start : chunk_start + batch_size]
        for offset, (sample_idx, sample) in enumerate(chunk):
            if cancel_callback and cancel_callback():
                return builder.samples, cancelled_batch_status(
                    success_count, total_to_label, total_to_label - processed_count
                )
            start_msg = start_batch_item(
                progress_tracker,
                progress_callback,
                position=sample_idx + 1,
                labeled_count=initial_labeled_count + success_count,
                left_count=total_to_label - chunk_start - offset,
                filename=sample.filename,
            )
            if requires_single_label_path(sample, format_lyrics=format_lyrics):
                status, sidecar_failed = label_single_fallback(
                    builder,
                    sample_idx,
                    dit_handler,
                    llm_handler,
                    format_lyrics,
                    transcribe_lyrics,
                    lm_lyrics_language,
                    skip_metas,
                    progress_callback,
                    start_msg,
                    persist_labels,
                    label_output_dir,
                    resolved_source_root,
                    sample_labeled_callback,
                )
                processed_count, success_count, fail_count, sidecar_fail_count = (
                    finish_label_counts(
                        builder.samples[sample_idx],
                        processed_count,
                        success_count,
                        fail_count,
                        sidecar_fail_count,
                        sidecar_failed,
                    )
                )
                complete_batch_item(
                    progress_tracker,
                    progress_callback,
                    position=sample_idx + 1,
                    labeled_count=initial_labeled_count + success_count,
                    left_count=total_to_label - processed_count,
                    filename=sample.filename,
                )
                _ = status
                continue

            if progress_callback:
                progress_callback(
                    f"Encoding {sample_idx + 1}/{len(builder.samples)}: {sample.filename}"
                )
            audio_codes = get_audio_codes(sample.audio_path, dit_handler)
            if not audio_codes:
                processed_count += 1
                fail_count += 1
                complete_batch_item(
                    progress_tracker,
                    progress_callback,
                    position=sample_idx + 1,
                    labeled_count=initial_labeled_count + success_count,
                    left_count=total_to_label - processed_count,
                    filename=sample.filename,
                )
                continue
            entries.append(_BatchEntry(sample_idx, sample.filename, audio_codes))

        if entries:
            if cancel_callback and cancel_callback():
                return builder.samples, cancelled_batch_status(
                    success_count, total_to_label, total_to_label - processed_count
                )
            metadata_results = generate_metadata_batch(
                llm_handler,
                [entry.audio_codes for entry in entries],
                transcribe_lyrics=transcribe_lyrics,
                lm_lyrics_language=lm_lyrics_language,
                progress_callback=progress_callback,
            )
            for entry, (metadata, status) in zip(entries, metadata_results):
                sidecar_failed = False
                if metadata:
                    sample, status = apply_understood_metadata(
                        builder.samples[entry.sample_idx],
                        metadata,
                        transcribe_lyrics=transcribe_lyrics,
                        lm_lyrics_language=lm_lyrics_language,
                        skip_metas=skip_metas,
                    )
                    builder.samples[entry.sample_idx] = sample
                    sidecar_failed = persist_successful_label(
                        sample,
                        entry.sample_idx,
                        status,
                        persist_labels,
                        label_output_dir,
                        resolved_source_root,
                        sample_labeled_callback,
                    )
                else:
                    builder.samples[entry.sample_idx].labeled = False
                    _ = status
                counts = finish_label_counts(
                    builder.samples[entry.sample_idx],
                    processed_count,
                    success_count,
                    fail_count,
                    sidecar_fail_count,
                    sidecar_failed,
                )
                processed_count, success_count, fail_count, sidecar_fail_count = counts
                complete_batch_item(
                    progress_tracker,
                    progress_callback,
                    position=entry.sample_idx + 1,
                    labeled_count=initial_labeled_count + success_count,
                    left_count=total_to_label - processed_count,
                    filename=entry.filename,
                )

    return builder.samples, final_batch_status(
        success_count,
        total_to_label,
        fail_count,
        sidecar_fail_count,
        only_unlabeled,
        skipped_count,
        len(builder.samples),
    )
