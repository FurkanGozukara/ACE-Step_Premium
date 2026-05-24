"""Tests for real batched dataset auto-labeling."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.training.dataset_builder_modules.label_all import LabelAllMixin
from acestep.training.dataset_builder_modules.models import AudioSample


class _Builder(LabelAllMixin):
    """Minimal builder for batched label-all tests."""

    def __init__(self, samples: list[AudioSample]) -> None:
        """Store samples for the mixin under test."""

        self.samples = samples

    def get_labeled_count(self) -> int:
        """Return the number of samples already labeled."""

        return sum(1 for sample in self.samples if sample.labeled)

    def label_sample(self, *_args, **_kwargs):
        """Fail if the batchable path accidentally falls back to single labels."""

        raise AssertionError("label_sample should not be used for batchable samples")


class _BatchLlm:
    """LLM test double exposing prompt-level batch generation."""

    def __init__(self) -> None:
        """Capture each prompt batch length."""

        self.batch_lengths: list[int] = []

    def build_formatted_prompt_for_understanding(self, audio_codes: str) -> str:
        """Return a prompt that keeps the source code visible to assertions."""

        return f"prompt:{audio_codes}"

    def generate_from_formatted_prompt(self, formatted_prompt, **_kwargs):
        """Return one synthetic LM output per prompt."""

        self.batch_lengths.append(len(formatted_prompt))
        return [f"output:{prompt}" for prompt in formatted_prompt], "ok"

    def parse_lm_output(self, output_text: str):
        """Return valid metadata for the encoded prompt."""

        code = output_text.split("prompt:", 1)[1]
        return {
            "caption": f"caption {code}",
            "genres": "electronic",
            "bpm": "120",
            "duration": "30",
            "keyscale": "C major",
            "language": "en",
            "timesignature": "4",
        }, ""

    def _extract_lyrics_from_output(self, _output_text: str) -> str:
        """Return usable lyrics so transcribed samples become non-instrumental."""

        return "[Verse]\nBatch lyrics"


class LabelBatchTests(unittest.TestCase):
    """Verify batch_size performs grouped LM work instead of being ignored."""

    @patch(
        "acestep.training.dataset_builder_modules.label_batch_persistence."
        "save_sample_label_metadata"
    )
    @patch(
        "acestep.training.dataset_builder_modules.label_batch.get_audio_codes",
        side_effect=lambda audio_path, _handler: f"codes-{audio_path}",
    )
    def test_batch_size_groups_understand_requests(self, _get_codes, save_sidecar) -> None:
        """Batchable samples should use real prompt batches and still persist labels."""

        builder = _Builder(
            [
                AudioSample(audio_path=f"{idx}.wav", filename=f"{idx}.wav")
                for idx in range(4)
            ]
        )
        llm_handler = _BatchLlm()
        callback_indexes: list[int] = []

        _samples, status = builder.label_all_samples(
            dit_handler=object(),
            llm_handler=llm_handler,
            transcribe_lyrics=True,
            lm_lyrics_language="en",
            batch_size=2,
            sample_labeled_callback=lambda idx, _sample, _status: callback_indexes.append(idx),
        )

        self.assertEqual([2, 2], llm_handler.batch_lengths)
        self.assertEqual([0, 1, 2, 3], callback_indexes)
        self.assertEqual(4, save_sidecar.call_count)
        self.assertIn("Labeled 4/4 samples; left 0", status)
        self.assertEqual("caption codes-0.wav", builder.samples[0].caption)
        self.assertFalse(builder.samples[0].is_instrumental)


if __name__ == "__main__":
    unittest.main()
