"""Tests for language-aware LoRA lyrics preprocessing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from acestep.training.dataset_builder_modules.preprocess_lyrics import (
    encode_lyrics,
    format_lyrics_input,
)


class PreprocessLyricsTests(unittest.TestCase):
    """Verify training receives the same lyrics envelope as inference."""

    def test_formats_dutch_language_code_without_translation(self) -> None:
        lyrics = "[Verse]\nDit is een Nederlandse regel"

        result = format_lyrics_input(lyrics, "nl")

        self.assertEqual(
            "# Languages\nnl\n\n"
            "# Lyric\n[Verse]\nDit is een Nederlandse regel<|endoftext|>",
            result,
        )

    def test_blank_language_uses_unknown(self) -> None:
        self.assertEqual(
            "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>",
            format_lyrics_input("", ""),
        )

    def test_encoder_uses_inference_lyrics_limit_without_fixed_padding(self) -> None:
        tokenizer = MagicMock(
            return_value=SimpleNamespace(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )
        )
        text_encoder = MagicMock()
        text_encoder.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )
        text_encoder.embed_tokens.return_value = torch.zeros(1, 3, 4)
        lyrics_input = format_lyrics_input("Nederlandse woorden", "nl")

        encode_lyrics(
            text_encoder,
            tokenizer,
            lyrics_input,
            torch.device("cpu"),
            torch.float32,
        )

        tokenizer.assert_called_once_with(
            lyrics_input,
            padding="longest",
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )


if __name__ == "__main__":
    unittest.main()
