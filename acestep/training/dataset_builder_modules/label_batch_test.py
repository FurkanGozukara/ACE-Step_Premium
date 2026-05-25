"""Tests for real batched dataset auto-labeling."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training.dataset_builder_modules.label_all import LabelAllMixin
from acestep.training.dataset_builder_modules.label_batch_apply import (
    apply_understood_metadata,
)
from acestep.training.dataset_builder_modules.models import AudioSample


class _Builder(LabelAllMixin):
    """Minimal builder for batched label-all tests."""

    def __init__(self, samples: list[AudioSample]) -> None:
        """Store samples for the mixin under test."""

        self.samples = samples
        self.metadata = SimpleNamespace(
            custom_tag="",
            tag_position="prepend",
            use_only_custom_trigger=False,
        )

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


class _SlowBatchLlm(_BatchLlm):
    """LLM test double that keeps the batch call open long enough for progress."""

    def generate_from_formatted_prompt(self, formatted_prompt, **kwargs):
        """Delay the synthetic generation call before returning outputs."""

        time.sleep(0.04)
        return super().generate_from_formatted_prompt(formatted_prompt, **kwargs)


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
                AudioSample(
                    audio_path="0.wav",
                    filename="0.wav",
                    caption="user caption from lyric file",
                    caption_source="lyrics_file",
                    raw_lyrics="[Verse]\nuser lyric",
                    lyrics="[Verse]\nuser lyric",
                    is_instrumental=False,
                ),
                *[
                    AudioSample(audio_path=f"{idx}.wav", filename=f"{idx}.wav")
                    for idx in range(1, 4)
                ],
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
        self.assertEqual("user caption from lyric file", builder.samples[0].caption)
        self.assertEqual("caption codes-1.wav", builder.samples[1].caption)
        self.assertFalse(builder.samples[0].is_instrumental)

    @patch(
        "acestep.training.dataset_builder_modules.label_batch_persistence."
        "save_sample_label_metadata"
    )
    @patch(
        "acestep.training.dataset_builder_modules.label_batch.get_audio_codes",
        side_effect=lambda audio_path, _handler: f"codes-{audio_path}",
    )
    @patch(
        "acestep.training.dataset_builder_modules.label_batch_generation."
        "METADATA_BATCH_HEARTBEAT_SECONDS",
        0.01,
    )
    def test_batch_metadata_generation_reports_heartbeat(
        self,
        _get_codes,
        _save_sidecar,
    ) -> None:
        """Batched metadata generation should emit elapsed progress while blocked."""

        builder = _Builder(
            [
                AudioSample(audio_path="0.wav", filename="0.wav"),
                AudioSample(audio_path="1.wav", filename="1.wav"),
            ]
        )
        progress_messages: list[str] = []

        builder.label_all_samples(
            dit_handler=object(),
            llm_handler=_SlowBatchLlm(),
            batch_size=2,
            progress_callback=progress_messages.append,
        )

        heartbeat_messages = [
            message for message in progress_messages if " | elapsed " in message
        ]
        self.assertTrue(
            any(
                message.startswith("Generating metadata batch (2 files)...")
                for message in heartbeat_messages
            )
        )

    @patch(
        "acestep.training.dataset_builder_modules.label_batch_persistence."
        "save_sample_label_metadata"
    )
    @patch(
        "acestep.training.dataset_builder_modules.label_batch.get_audio_codes",
        side_effect=lambda audio_path, _handler: f"codes-{audio_path}",
    )
    def test_batch_use_only_custom_trigger_overwrites_captions(
        self,
        _get_codes,
        save_sidecar,
    ) -> None:
        """Batch labeling should persist only the trigger when requested."""

        builder = _Builder(
            [
                AudioSample(audio_path="0.wav", filename="0.wav"),
                AudioSample(audio_path="1.wav", filename="1.wav"),
            ]
        )
        builder.metadata.custom_tag = "ohwx"

        _samples, status = builder.label_all_samples(
            dit_handler=object(),
            llm_handler=_BatchLlm(),
            batch_size=2,
            use_only_custom_trigger=True,
        )

        self.assertIn("Labeled 2/2 samples; left 0", status)
        self.assertEqual(["ohwx", "ohwx"], [sample.caption for sample in builder.samples])
        self.assertEqual(["", ""], [sample.custom_tag for sample in builder.samples])
        self.assertEqual("replace", builder.metadata.tag_position)
        self.assertTrue(builder.metadata.use_only_custom_trigger)
        self.assertEqual("ohwx", save_sidecar.call_args_list[0].args[0].caption)
        self.assertEqual("", save_sidecar.call_args_list[0].args[0].custom_tag)

    def test_batch_empty_transcription_keeps_vocal_metadata_non_instrumental(self) -> None:
        """Batched auto-labeling should not mark clear vocal metadata instrumental."""

        sample = AudioSample(
            audio_path="song.wav",
            filename="song.wav",
            is_instrumental=False,
        )
        metadata = {
            "caption": "Multiple male rappers deliver confident rhythmic verses.",
            "genres": "G-funk",
            "lyrics": "",
            "language": "en",
        }

        sample, status = apply_understood_metadata(
            sample,
            metadata,
            transcribe_lyrics=True,
            lm_lyrics_language="en",
            skip_metas=False,
        )

        self.assertFalse(sample.is_instrumental)
        self.assertEqual("[Instrumental]", sample.lyrics)
        self.assertEqual("", sample.formatted_lyrics)
        self.assertIn("vocal metadata", status)


if __name__ == "__main__":
    unittest.main()
