"""Tests for single-sample dataset auto-labeling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.training.dataset_builder_modules.label_single import LabelSingleMixin
from acestep.training.dataset_builder_modules.models import AudioSample


class _Builder(LabelSingleMixin):
    """Minimal builder exposing samples for ``LabelSingleMixin`` tests."""

    def __init__(self, samples: list[AudioSample]) -> None:
        """Store samples for the mixin under test."""

        self.samples = samples


class LabelSingleMixinTests(unittest.TestCase):
    """Coverage for LM lyric format/transcription paths."""

    def test_transcribe_lyrics_overrides_default_instrumental_flag(self) -> None:
        """Explicit transcription should not be suppressed by the scan default."""

        builder = _Builder(
            [AudioSample(audio_path="song.wav", filename="song.wav", is_instrumental=True)]
        )
        llm_handler = MagicMock()
        llm_handler.understand_audio_from_codes.return_value = (
            {
                "caption": "hip hop beat with vocals",
                "genres": "hip hop",
                "lyrics": "[Verse]\nhello from the hook",
                "language": "en",
            },
            "ok",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=llm_handler,
                transcribe_lyrics=True,
                lm_lyrics_language="en",
            )

        self.assertIs(sample, builder.samples[0])
        self.assertFalse(sample.is_instrumental)
        self.assertEqual("en", sample.language)
        self.assertEqual("[Verse]\nhello from the hook", sample.lyrics)
        self.assertEqual(sample.lyrics, sample.formatted_lyrics)
        self.assertIn("lyrics transcribed by LM", status)
        llm_handler.understand_audio_from_codes.assert_called_once_with(
            audio_codes="<|audio_code_1|>",
            temperature=0.1,
            top_p=0.3,
            user_metadata={"language": "en"},
            use_constrained_decoding=True,
        )

    def test_transcribe_lyrics_preserves_preloaded_file_when_format_disabled(self) -> None:
        """User-provided lyrics should win over LM transcription output."""

        raw_lyrics = "[Verse 1]\nsource lyric from the formatted file"
        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    raw_lyrics=raw_lyrics,
                    lyrics=raw_lyrics,
                    is_instrumental=False,
                )
            ]
        )
        llm_handler = MagicMock()
        llm_handler.understand_audio_from_codes.return_value = (
            {
                "caption": "audio-inferred caption",
                "genres": "west coast hip hop",
                "lyrics": "[Verse]\nhallucinated transcript",
                "language": "en",
            },
            "ok",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=llm_handler,
                format_lyrics=False,
                transcribe_lyrics=True,
                lm_lyrics_language="en",
            )

        self.assertEqual("audio-inferred caption", sample.caption)
        self.assertEqual("west coast hip hop", sample.genre)
        self.assertFalse(sample.is_instrumental)
        self.assertEqual(raw_lyrics, sample.lyrics)
        self.assertEqual("", sample.formatted_lyrics)
        self.assertNotIn("hallucinated", sample.lyrics)
        self.assertIn("using raw lyrics", status)

    def test_lyric_file_caption_overrides_lm_metadata_caption(self) -> None:
        """A caption from a lyric sidecar should survive audio metadata generation."""

        raw_lyrics = "[Verse]\nsource lyric from the formatted file"
        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    caption="user caption from lyric file",
                    caption_source="lyrics_file",
                    raw_lyrics=raw_lyrics,
                    lyrics=raw_lyrics,
                    is_instrumental=False,
                )
            ]
        )
        llm_handler = MagicMock()
        llm_handler.understand_audio_from_codes.return_value = (
            {
                "caption": "model generated caption",
                "genres": "west coast hip hop",
                "bpm": "92",
                "keyscale": "G minor",
                "timesignature": "4",
                "language": "en",
            },
            "ok",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, _status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=llm_handler,
            )

        self.assertEqual("user caption from lyric file", sample.caption)
        self.assertEqual("west coast hip hop", sample.genre)
        self.assertEqual(92, sample.bpm)
        self.assertEqual(raw_lyrics, sample.lyrics)

    @patch("acestep.inference.format_sample")
    def test_format_lyrics_uses_lyric_file_caption_as_input(self, format_sample) -> None:
        """Lyric-sidecar captions should guide formatting and remain the final caption."""

        raw_lyrics = "hello world\nsing it loud"
        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    caption="user caption from lyric file",
                    caption_source="lyrics_file",
                    raw_lyrics=raw_lyrics,
                    lyrics=raw_lyrics,
                    is_instrumental=False,
                )
            ]
        )
        format_sample.return_value = SimpleNamespace(
            success=True,
            error="",
            caption="model generated caption",
            lyrics="[Verse]\nhello world\nsing it loud",
            bpm=120,
            keyscale="C major",
            timesignature="4",
            language="en",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, _status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=MagicMock(),
                format_lyrics=True,
            )

        self.assertEqual("user caption from lyric file", format_sample.call_args.kwargs["caption"])
        self.assertEqual("user caption from lyric file", sample.caption)
        self.assertEqual("[Verse]\nhello world\nsing it loud", sample.lyrics)

    def test_instrumental_default_still_applies_without_transcription(self) -> None:
        """Existing instrumental behavior should remain unchanged by default."""

        builder = _Builder(
            [AudioSample(audio_path="song.wav", filename="song.wav", is_instrumental=True)]
        )
        llm_handler = MagicMock()
        llm_handler.understand_audio_from_codes.return_value = (
            {
                "caption": "hip hop beat with vocals",
                "genres": "hip hop",
                "lyrics": "[Verse]\nignored lyric",
                "language": "en",
            },
            "ok",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=llm_handler,
            )

        self.assertTrue(sample.is_instrumental)
        self.assertEqual("[Instrumental]", sample.lyrics)
        self.assertEqual("", sample.formatted_lyrics)
        self.assertIn("instrumental", status)

    def test_transcribe_lyrics_rejects_repetitive_output_without_raw_lyrics(self) -> None:
        """Repetitive transcription output should not be accepted as training lyrics."""

        builder = _Builder(
            [AudioSample(audio_path="song.wav", filename="song.wav", is_instrumental=True)]
        )
        llm_handler = MagicMock()
        llm_handler.understand_audio_from_codes.return_value = (
            {
                "caption": "hip hop beat with vocals",
                "genres": "hip hop",
                "lyrics": "[Verse]\n" + " ".join(["yeah"] * 40),
                "language": "en",
            },
            "ok",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=llm_handler,
                transcribe_lyrics=True,
                lm_lyrics_language="en",
            )

        self.assertTrue(sample.is_instrumental)
        self.assertEqual("[Instrumental]", sample.lyrics)
        self.assertEqual("", sample.formatted_lyrics)
        self.assertIn("LM transcription rejected", status)

    @patch("acestep.inference.format_sample")
    def test_format_lyrics_uses_raw_lyrics_file_content(self, format_sample) -> None:
        """Raw lyric files should be formatted and preserved in formatted_lyrics."""

        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    raw_lyrics="hello world\nsing it loud",
                    lyrics="hello world\nsing it loud",
                    is_instrumental=False,
                )
            ]
        )
        format_sample.return_value = SimpleNamespace(
            success=True,
            error="",
            caption="formatted caption",
            lyrics="[Verse]\nhello world\nsing it loud",
            bpm=120,
            keyscale="C major",
            timesignature="4",
            language="en",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=MagicMock(),
                format_lyrics=True,
                lm_lyrics_language="en",
            )

        format_sample.assert_called_once()
        self.assertEqual({"language": "en"}, format_sample.call_args.kwargs["user_metadata"])
        self.assertEqual(0.20, format_sample.call_args.kwargs["temperature"])
        self.assertEqual(0.75, format_sample.call_args.kwargs["top_p"])
        self.assertEqual(1.18, format_sample.call_args.kwargs["repetition_penalty"])
        self.assertFalse(sample.is_instrumental)
        self.assertEqual("[Verse]\nhello world\nsing it loud", sample.lyrics)
        self.assertEqual(sample.lyrics, sample.formatted_lyrics)
        self.assertIn("lyrics formatted by LM", status)

    @patch("acestep.inference.format_sample")
    def test_combined_format_and_transcribe_uses_file_lyrics_and_audio_metadata(
        self,
        format_sample,
    ) -> None:
        """Combined mode should format lyric files but infer metadata from audio."""

        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    raw_lyrics="real lyric line",
                    lyrics="real lyric line",
                    is_instrumental=False,
                )
            ]
        )
        format_sample.return_value = SimpleNamespace(
            success=True,
            error="",
            caption="lyric-only caption",
            lyrics="[Verse]\nreal lyric line",
            bpm=999,
            keyscale="D minor",
            timesignature="3",
            language="en",
        )
        llm_handler = MagicMock()
        llm_handler.understand_audio_from_codes.return_value = (
            {
                "caption": "audio-inferred caption",
                "genres": "melodic rap",
                "lyrics": "[Verse]\nhallucinated line",
                "bpm": "81",
                "keyscale": "C major",
                "timesignature": "4",
                "language": "en",
            },
            "ok",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=llm_handler,
                format_lyrics=True,
                transcribe_lyrics=True,
                lm_lyrics_language="unknown",
            )

        self.assertEqual("audio-inferred caption", sample.caption)
        self.assertEqual("melodic rap", sample.genre)
        self.assertEqual(81, sample.bpm)
        self.assertEqual("C major", sample.keyscale)
        self.assertEqual("4", sample.timesignature)
        self.assertEqual("en", sample.language)
        self.assertFalse(sample.is_instrumental)
        self.assertEqual("[Verse]\nreal lyric line", sample.lyrics)
        self.assertEqual(sample.lyrics, sample.formatted_lyrics)
        self.assertNotIn("hallucinated", sample.lyrics)
        self.assertIn("lyrics from file", status)
        self.assertIn("metadata inferred from audio", status)
        self.assertEqual("audio-inferred caption", format_sample.call_args.kwargs["caption"])
        self.assertEqual({"language": "en"}, format_sample.call_args.kwargs["user_metadata"])
        self.assertEqual(0.20, format_sample.call_args.kwargs["temperature"])
        self.assertEqual(0.75, format_sample.call_args.kwargs["top_p"])
        self.assertEqual(1.18, format_sample.call_args.kwargs["repetition_penalty"])
        llm_handler.understand_audio_from_codes.assert_called_once_with(
            audio_codes="<|audio_code_1|>",
            temperature=0.1,
            top_p=0.3,
            user_metadata=None,
            use_constrained_decoding=True,
        )

    @patch("acestep.inference.format_sample")
    def test_format_lyrics_preserves_raw_when_lm_returns_instrumental(
        self,
        format_sample,
    ) -> None:
        """Formatting should not overwrite raw lyric files with instrumental text."""

        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    raw_lyrics="hello world\nsing it loud",
                    lyrics="hello world\nsing it loud",
                    is_instrumental=False,
                )
            ]
        )
        format_sample.return_value = SimpleNamespace(
            success=True,
            error="",
            caption="formatted caption",
            lyrics="[Instrumental]",
            bpm=120,
            keyscale="C major",
            timesignature="4",
            language="unknown",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=MagicMock(),
                format_lyrics=True,
            )

        self.assertFalse(sample.is_instrumental)
        self.assertEqual("hello world\nsing it loud", sample.lyrics)
        self.assertEqual("", sample.formatted_lyrics)
        self.assertIn("using raw lyrics", status)

    @patch("acestep.inference.format_sample")
    def test_format_lyrics_preserves_raw_when_lm_returns_repetitive_text(
        self,
        format_sample,
    ) -> None:
        """Formatting should reject repetitive hallucinations and keep raw lyrics."""

        raw_lyrics = "\n".join(
            [
                "first real line from the source",
                "second real line from the source",
                "third real line from the source",
                "fourth real line from the source",
                "fifth real line from the source",
                "sixth real line from the source",
                "seventh real line from the source",
                "eighth real line from the source",
            ]
        )
        repeated_lyrics = "\n".join(
            [
                "[Intro]",
                "same generic hook",
                "same generic hook",
                "same generic hook",
                "same generic hook",
                "[Verse 1]",
                "same generic hook",
                "same generic hook",
                "same generic hook",
                "same generic hook",
                "[Chorus]",
                "same generic hook",
                "same generic hook",
                "same generic hook",
                "same generic hook",
            ]
        )
        builder = _Builder(
            [
                AudioSample(
                    audio_path="song.wav",
                    filename="song.wav",
                    raw_lyrics=raw_lyrics,
                    lyrics=raw_lyrics,
                    is_instrumental=False,
                )
            ]
        )
        format_sample.return_value = SimpleNamespace(
            success=True,
            error="",
            caption="formatted caption",
            lyrics=repeated_lyrics,
            bpm=120,
            keyscale="C major",
            timesignature="4",
            language="en",
        )

        with patch(
            "acestep.training.dataset_builder_modules.label_single.get_audio_codes",
            return_value="<|audio_code_1|>",
        ):
            sample, status = builder.label_sample(
                0,
                dit_handler=MagicMock(),
                llm_handler=MagicMock(),
                format_lyrics=True,
                lm_lyrics_language="en",
            )

        self.assertIn("[Intro]", sample.lyrics)
        self.assertIn("first real line from the source", sample.lyrics)
        self.assertEqual("", sample.formatted_lyrics)
        self.assertIn("LM format rejected", status)
        self.assertIn("repetitive formatted lyrics", status)


if __name__ == "__main__":
    unittest.main()
