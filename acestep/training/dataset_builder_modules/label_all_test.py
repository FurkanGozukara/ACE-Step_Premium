"""Tests for batch dataset auto-labeling behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training.dataset_builder_modules.label_all import LabelAllMixin
from acestep.training.dataset_builder_modules.models import AudioSample


class _Builder(LabelAllMixin):
    """Minimal builder implementing the methods used by ``LabelAllMixin``."""

    def __init__(self, samples: list[AudioSample]) -> None:
        """Store samples and record label calls."""

        self.samples = samples
        self.labeled_indexes: list[int] = []
        self.post_load_messages: list[str] = []

    def get_labeled_count(self) -> int:
        """Return the number of samples already marked labeled."""

        return sum(1 for sample in self.samples if sample.labeled)

    def label_sample(self, sample_idx: int, *_args, **_kwargs):
        """Mark a sample labeled without invoking model dependencies."""

        if len(_args) >= 2:
            message = getattr(_args[1], "_post_load_status_message", None)
            if message:
                self.post_load_messages.append(message)
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
        self.assertTrue(any("ETA" in message for message in progress_messages))

    @patch("acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata")
    def test_cancel_callback_stops_before_next_sample(self, save_sidecar) -> None:
        """Auto-label cancellation should stop the local loop between files."""

        builder = _Builder(
            [
                AudioSample(audio_path="a.wav", filename="a.wav"),
                AudioSample(audio_path="b.wav", filename="b.wav"),
            ]
        )
        checks = 0

        def cancel_after_first_check() -> bool:
            nonlocal checks
            checks += 1
            return checks > 1

        _samples, status = builder.label_all_samples(
            dit_handler=None,
            llm_handler=None,
            cancel_callback=cancel_after_first_check,
        )

        self.assertEqual([0], builder.labeled_indexes)
        self.assertEqual(1, save_sidecar.call_count)
        self.assertIn("Auto-label cancelled after 1/2 samples; left 1", status)

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
        progress_messages: list[str] = []

        _samples, status = builder.label_all_samples(
            dit_handler=None,
            llm_handler=None,
            progress_callback=progress_messages.append,
        )

        self.assertEqual([1], builder.labeled_indexes)
        self.assertEqual(1, save_sidecar.call_count)
        self.assertIn("1 already labeled", status)
        self.assertTrue(
            any(
                message.startswith("Labeling 2/2; labeled 1/2; left 1:")
                for message in progress_messages
            )
        )
        self.assertTrue(
            any(
                message.startswith("Labeling 2/2 complete; labeled 2/2; left 0:")
                for message in progress_messages
            )
        )

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

    @patch("acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata")
    def test_replays_current_progress_after_llm_load(self, _save_sidecar) -> None:
        """The LLM load hook should re-emit the current file progress line."""

        builder = _Builder([AudioSample(audio_path="todo.wav", filename="todo.wav")])
        previous_message = "previous message"
        llm_handler = SimpleNamespace(_post_load_status_message=previous_message)
        progress_messages: list[str] = []

        builder.label_all_samples(
            dit_handler=None,
            llm_handler=llm_handler,
            progress_callback=progress_messages.append,
        )

        current_lines = [
            message
            for message in builder.post_load_messages
            if message.startswith("Labeling 1/1; labeled 0/1; left 1:")
        ]
        self.assertEqual(1, len(current_lines))
        self.assertEqual(previous_message, llm_handler._post_load_status_message)

    @patch("acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata")
    def test_passes_processed_label_folder_to_persistence(self, save_sidecar) -> None:
        """Batch labeling should persist labels to the selected processed folder."""

        builder = _Builder([AudioSample(audio_path="todo.wav", filename="todo.wav")])

        _samples, status = builder.label_all_samples(
            dit_handler=None,
            llm_handler=None,
            label_output_dir="labels",
            label_source_root="source",
        )

        self.assertIn("Labeled 1/1", status)
        save_sidecar.assert_called_once()
        self.assertEqual("labels", save_sidecar.call_args.kwargs["output_dir"])
        self.assertEqual("source", save_sidecar.call_args.kwargs["source_root"])

    @patch(
        "acestep.training.dataset_builder_modules.label_all.save_sample_label_metadata",
        side_effect=OSError("read-only dataset directory"),
    )
    @patch("acestep.training.dataset_builder_modules.label_all.logger.exception")
    def test_reports_sidecar_save_failures_in_final_status(
        self,
        _logger_exception,
        _save_sidecar,
    ) -> None:
        """Sidecar write failures should not be hidden in logs only."""

        builder = _Builder([AudioSample(audio_path="todo.wav", filename="todo.wav")])

        _samples, status = builder.label_all_samples(dit_handler=None, llm_handler=None)

        self.assertIn("Labeled 1/1", status)
        self.assertIn("1 sidecar save failed", status)


if __name__ == "__main__":
    unittest.main()
