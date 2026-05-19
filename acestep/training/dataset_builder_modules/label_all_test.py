"""Tests for batch dataset auto-labeling behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.training.dataset_builder_modules.label_all import LabelAllMixin
from acestep.training.dataset_builder_modules.models import AudioSample


class _Builder(LabelAllMixin):
    """Minimal builder implementing the methods used by ``LabelAllMixin``."""

    def __init__(self, samples: list[AudioSample]) -> None:
        """Store samples and record label calls."""

        self.samples = samples
        self.labeled_indexes: list[int] = []

    def get_labeled_count(self) -> int:
        """Return the number of samples already marked labeled."""

        return sum(1 for sample in self.samples if sample.labeled)

    def label_sample(self, sample_idx: int, *_args, **_kwargs):
        """Mark a sample labeled without invoking model dependencies."""

        sample = self.samples[sample_idx]
        sample.caption = sample.caption or f"caption {sample_idx}"
        sample.labeled = True
        self.samples[sample_idx] = sample
        self.labeled_indexes.append(sample_idx)
        return sample, "\u2705 Labeled"


class LabelAllMixinTests(unittest.TestCase):
    """Regression coverage for auto-label progress, skip, and persistence callbacks."""

    @patch("acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata")
    def test_labels_unlabeled_samples_and_reports_counts(self, save_sidecar) -> None:
        """Successful labels should be persisted and reported through callbacks."""

        builder = _Builder(
            [
                AudioSample(audio_path="a.wav", filename="a.wav"),
                AudioSample(audio_path="b.wav", filename="b.wav"),
            ]
        )
        progress_messages: list[str] = []
        callback_indexes: list[int] = []

        _samples, status = builder.label_all_samples(
            dit_handler=None,
            llm_handler=None,
            progress_callback=progress_messages.append,
            sample_labeled_callback=lambda idx, _sample, _status: callback_indexes.append(idx),
        )

        self.assertEqual([0, 1], builder.labeled_indexes)
        self.assertEqual([0, 1], callback_indexes)
        self.assertEqual(2, save_sidecar.call_count)
        self.assertIn("Labeled 2/2 samples; left 0", status)
        self.assertTrue(any("left 2" in message for message in progress_messages))
        self.assertTrue(any("left 0" in message for message in progress_messages))

    @patch("acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata")
    def test_skips_already_labeled_samples_by_default(self, save_sidecar) -> None:
        """Already labeled samples should not be relabeled on continuation runs."""

        builder = _Builder(
            [
                AudioSample(
                    audio_path="done.wav",
                    filename="done.wav",
                    caption="existing caption",
                    labeled=True,
                ),
                AudioSample(audio_path="todo.wav", filename="todo.wav"),
            ]
        )

        _samples, status = builder.label_all_samples(dit_handler=None, llm_handler=None)

        self.assertEqual([1], builder.labeled_indexes)
        self.assertEqual(1, save_sidecar.call_count)
        self.assertIn("1 already labeled", status)

    def test_returns_complete_status_when_all_samples_are_labeled(self) -> None:
        """All-labeled datasets should finish without invoking label_sample."""

        builder = _Builder(
            [
                AudioSample(
                    audio_path="done.wav",
                    filename="done.wav",
                    caption="existing caption",
                    labeled=True,
                )
            ]
        )

        _samples, status = builder.label_all_samples(dit_handler=None, llm_handler=None)

        self.assertEqual([], builder.labeled_indexes)
        self.assertIn("All samples already labeled", status)

    def test_accepts_api_batch_arguments(self) -> None:
        """API batch/chunk arguments should not break the local label loop."""

        builder = _Builder([AudioSample(audio_path="todo.wav", filename="todo.wav")])

        with patch("acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata"):
            _samples, status = builder.label_all_samples(
                dit_handler=None,
                llm_handler=None,
                chunk_size=16,
                batch_size=1,
            )

        self.assertIn("Labeled 1/1", status)


if __name__ == "__main__":
    unittest.main()
