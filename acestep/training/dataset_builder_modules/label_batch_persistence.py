"""Persistence and accounting helpers for batched auto-labeling."""

from __future__ import annotations

import os
from typing import Callable

from loguru import logger

from acestep.training.path_safety import safe_path

from .label_persistence import save_sample_label_metadata
from .models import AudioSample


_SUCCESS = "\u2705"
_WARNING = "\u26a0\ufe0f"


def persist_successful_label(
    sample: AudioSample,
    sample_idx: int,
    status: str,
    persist_labels: bool,
    label_output_dir: str | None,
    label_source_root: str | None,
    sample_labeled_callback: Callable[[int, AudioSample, str], None] | None,
) -> bool:
    """Persist one successful sample and run the incremental callback."""

    if not (sample.labeled and sample.caption):
        return False
    sidecar_failed = False
    if persist_labels:
        try:
            save_sample_label_metadata(
                sample,
                output_dir=label_output_dir,
                source_root=label_source_root,
            )
        except Exception as exc:
            sidecar_failed = True
            logger.exception("Auto-label sidecar save failed")
            status = f"{status}\n{_WARNING} Sidecar save failed: {exc}"
    if sample_labeled_callback:
        sample_labeled_callback(sample_idx, sample, status)
    return sidecar_failed


def finish_label_counts(
    sample: AudioSample,
    processed_count: int,
    success_count: int,
    fail_count: int,
    sidecar_fail_count: int,
    sidecar_failed: bool,
) -> tuple[int, int, int, int]:
    """Return updated processed, success, failure, and sidecar counters."""

    processed_count += 1
    if sample.labeled and sample.caption:
        success_count += 1
    else:
        fail_count += 1
    if sidecar_failed:
        sidecar_fail_count += 1
    return processed_count, success_count, fail_count, sidecar_fail_count


def final_batch_status(
    success_count: int,
    total_to_label: int,
    fail_count: int,
    sidecar_fail_count: int,
    only_unlabeled: bool,
    skipped_count: int,
    total_samples: int,
) -> str:
    """Build the final batch auto-label status line."""

    status_msg = f"{_SUCCESS} Labeled {success_count}/{total_to_label} samples; left 0"
    if fail_count > 0:
        status_msg += f" ({fail_count} failed)"
    if sidecar_fail_count > 0:
        status_msg += f" ({sidecar_fail_count} sidecar save failed)"
    if only_unlabeled:
        status_msg += f" ({skipped_count} already labeled, {total_samples} total)"
    return status_msg


def cancelled_batch_status(
    success_count: int,
    total_to_label: int,
    left_count: int,
) -> str:
    """Build the final status line for a cancelled batch auto-label run."""

    return (
        f"{_WARNING} Auto-label cancelled after {success_count}/{total_to_label} "
        f"samples; left {left_count}"
    )


def common_audio_source_root(samples: list[AudioSample]) -> str | None:
    """Return the common source-audio directory for processed-label naming."""

    directories: list[str] = []
    for sample in samples:
        try:
            directories.append(os.path.dirname(safe_path(sample.audio_path)))
        except (OSError, ValueError):
            continue
    if not directories:
        return None
    try:
        return os.path.commonpath(directories)
    except ValueError:
        return None
