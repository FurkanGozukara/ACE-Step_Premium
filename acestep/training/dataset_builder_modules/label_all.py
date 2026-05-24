"""Batch auto-labeling helpers for dataset samples."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from loguru import logger

from acestep.training.path_safety import safe_path

from .label_batch import label_samples_in_batches
from .label_batch_progress import normalize_auto_label_batch_size
from .label_persistence import save_sample_label_metadata
from .label_progress import LabelProgressTracker, replay_progress_after_llm_load
from .models import AudioSample


_SUCCESS = "\u2705"
_FAILURE = "\u274c"
_WARNING = "\u26a0\ufe0f"
_CANCELLED = "\u26a0\ufe0f"


def _needs_label(sample: AudioSample) -> bool:
    """Return whether a sample still needs auto-label work."""

    return not sample.labeled or not bool(sample.caption and sample.caption.strip())


class LabelAllMixin:
    """Label all samples in the dataset."""

    def label_all_samples(
        self,
        dit_handler,
        llm_handler,
        format_lyrics: bool = False,
        transcribe_lyrics: bool = False,
        lm_lyrics_language: str = "unknown",
        skip_metas: bool = False,
        only_unlabeled: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
        chunk_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        sample_labeled_callback: Optional[Callable[[int, AudioSample, str], None]] = None,
        persist_labels: bool = True,
        label_output_dir: str | None = None,
        label_source_root: str | None = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> Tuple[List[AudioSample], str]:
        """Label samples and persist each successful label immediately."""

        _ = chunk_size
        resolved_batch_size = normalize_auto_label_batch_size(batch_size)
        if not self.samples:
            return [], f"{_FAILURE} No samples to label. Please scan a directory first."

        if only_unlabeled:
            samples_to_label = [
                (i, sample) for i, sample in enumerate(self.samples) if _needs_label(sample)
            ]
        else:
            samples_to_label = [(i, sample) for i, sample in enumerate(self.samples)]

        if not samples_to_label:
            labeled_count = self.get_labeled_count()
            status = f"{_SUCCESS} All samples already labeled ({labeled_count} labeled, 0 left)"
            return self.samples, status

        initial_labeled_count = self.get_labeled_count() if only_unlabeled else 0
        if resolved_batch_size > 1:
            return label_samples_in_batches(
                self,
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                samples_to_label=samples_to_label,
                batch_size=resolved_batch_size,
                format_lyrics=format_lyrics,
                transcribe_lyrics=transcribe_lyrics,
                lm_lyrics_language=lm_lyrics_language,
                skip_metas=skip_metas,
                only_unlabeled=only_unlabeled,
                progress_callback=progress_callback,
                sample_labeled_callback=sample_labeled_callback,
                persist_labels=persist_labels,
                label_output_dir=label_output_dir,
                label_source_root=label_source_root,
                initial_labeled_count=initial_labeled_count,
                cancel_callback=cancel_callback,
            )

        success_count = 0
        fail_count = 0
        sidecar_fail_count = 0
        total_to_label = len(samples_to_label)
        display_total = len(self.samples)
        skipped_count = len(self.samples) - total_to_label if only_unlabeled else 0
        resolved_label_source_root = (
            label_source_root
            or getattr(self, "_current_dir", None)
            or _common_audio_source_root(self.samples)
        )
        progress_tracker = LabelProgressTracker(display_total)

        for idx, (sample_idx, sample) in enumerate(samples_to_label):
            if cancel_callback and cancel_callback():
                left_count = total_to_label - idx
                return self.samples, _cancelled_status(success_count, total_to_label, left_count)
            labeled_before = initial_labeled_count + success_count
            left_before = total_to_label - idx
            progress_tracker.begin_item()
            start_msg = progress_tracker.start_message(
                sample_idx + 1,
                labeled_before,
                left_before,
                sample.filename,
            )
            if progress_callback:
                progress_callback(start_msg)

            with replay_progress_after_llm_load(llm_handler, progress_callback, start_msg):
                _, status = self.label_sample(
                    sample_idx,
                    dit_handler,
                    llm_handler,
                    format_lyrics,
                    transcribe_lyrics,
                    lm_lyrics_language,
                    skip_metas,
                    progress_callback,
                )

            sample = self.samples[sample_idx]
            if sample.labeled and sample.caption:
                success_count += 1
                if persist_labels:
                    try:
                        save_sample_label_metadata(
                            sample,
                            output_dir=label_output_dir,
                            source_root=resolved_label_source_root,
                        )
                    except Exception as exc:
                        sidecar_fail_count += 1
                        logger.exception("Auto-label sidecar save failed")
                        status = f"{status}\n{_WARNING} Sidecar save failed: {exc}"
                if sample_labeled_callback:
                    sample_labeled_callback(sample_idx, sample, status)
            else:
                fail_count += 1

            left_after = total_to_label - idx - 1
            progress_tracker.complete_item()
            if progress_callback:
                labeled_after = initial_labeled_count + success_count
                progress_callback(
                    progress_tracker.complete_message(
                        sample_idx + 1,
                        labeled_after,
                        left_after,
                        sample.filename,
                    )
                )

        status_msg = f"{_SUCCESS} Labeled {success_count}/{total_to_label} samples; left 0"
        if fail_count > 0:
            status_msg += f" ({fail_count} failed)"
        if sidecar_fail_count > 0:
            status_msg += f" ({sidecar_fail_count} sidecar save failed)"
        if only_unlabeled:
            status_msg += f" ({skipped_count} already labeled, {len(self.samples)} total)"

        return self.samples, status_msg


def _cancelled_status(success_count: int, total_to_label: int, left_count: int) -> str:
    """Build the status line for a user-cancelled auto-label run."""

    return (
        f"{_CANCELLED} Auto-label cancelled after {success_count}/{total_to_label} "
        f"samples; left {left_count}"
    )


def _common_audio_source_root(samples: list[AudioSample]) -> str | None:
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
