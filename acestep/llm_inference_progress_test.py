"""Tests for LLM inference console-progress behavior."""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    from acestep.llm_inference import LLMHandler

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - dependency guard
    LLMHandler = None
    _IMPORT_ERROR = exc


class _FakeModel:
    """Minimal model object for exercising the load context."""

    def __init__(self) -> None:
        """Store calls made to ``to``."""

        self.to_calls: list[object] = []

    def parameters(self):
        """Return one CPU parameter-like object."""

        return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

    def to(self, target):
        """Record model moves and return self for chained calls."""

        self.to_calls.append(target)
        return self


class _FakeBatchTokenizer:
    """Tokenizer double that supports left-padded prompt batches."""

    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        """Store padding configuration and decode calls."""

        self.padding_side = "right"

    def __call__(self, texts, return_tensors, padding, truncation):
        """Tokenize strings into deterministic tensor lengths."""

        _ = return_tensors, truncation
        text_list = texts if isinstance(texts, list) else [texts]
        encoded = [[index + 10] * len(text) for index, text in enumerate(text_list)]
        max_len = max(len(tokens) for tokens in encoded)
        input_ids = []
        attention_mask = []
        for tokens in encoded:
            pad_count = max_len - len(tokens)
            if padding and self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_count + tokens)
                attention_mask.append([0] * pad_count + [1] * len(tokens))
            else:
                input_ids.append(tokens + [self.pad_token_id] * pad_count)
                attention_mask.append([1] * len(tokens) + [0] * pad_count)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        """Return visible token IDs for assertions."""

        _ = skip_special_tokens
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return "|".join(str(value) for value in values)


class _FakeBatchModel:
    """Model double that records native batch generate calls."""

    def __init__(self) -> None:
        """Store generated call shapes."""

        self.config = SimpleNamespace(max_new_tokens=3)
        self.generate_shapes: list[tuple[int, int]] = []

    def generate(self, **kwargs):
        """Append fixed row-specific tokens to the padded inputs."""

        input_ids = kwargs["input_ids"]
        self.generate_shapes.append(tuple(input_ids.shape))
        suffix = torch.tensor(
            [
                [101, 99, 0],
                [201, 202, 99],
            ],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix[: input_ids.shape[0]]], dim=1)


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmInferenceProgressTests(unittest.TestCase):
    """Verify LLM console progress hooks remain quiet and ordered."""

    def test_load_model_context_prints_post_load_status_message(self) -> None:
        """A configured status message should print after loading the model."""

        handler = LLMHandler()
        model = _FakeModel()
        handler.llm = model
        handler.llm_backend = "pt"
        handler.offload_to_cpu = True
        handler.device = "cuda"
        handler._post_load_status_message = (
            "Labeling 4/50; labeled 3/50; left 47: song.flac"
        )

        stderr = io.StringIO()
        with patch("torch.cuda.is_available", return_value=False):
            with contextlib.redirect_stderr(stderr):
                with handler._load_model_context():
                    pass

        self.assertIn(
            "Labeling 4/50; labeled 3/50; left 47: song.flac",
            stderr.getvalue(),
        )
        self.assertEqual("cuda", model.to_calls[0])
        self.assertEqual("cpu", model.to_calls[-1])

    def test_understand_audio_from_codes_does_not_print_full_prompt(self) -> None:
        """Understanding audio codes should not dump code tokens to stdout."""

        handler = LLMHandler()
        handler.llm_initialized = True
        audio_codes = "<|audio_code_1|><|audio_code_2|>"

        with patch.object(
            handler,
            "build_formatted_prompt_for_understanding",
            return_value=f"prompt {audio_codes}",
        ), patch.object(
            handler,
            "generate_from_formatted_prompt",
            return_value=("<think></think>", "ok"),
        ), patch.object(
            handler,
            "parse_lm_output",
            return_value=({}, ""),
        ):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                metadata, status = handler.understand_audio_from_codes(audio_codes)

        self.assertEqual({}, metadata)
        self.assertIn("Understanding completed", status)
        self.assertEqual("", stdout.getvalue())

    def test_pt_unconstrained_prompt_batch_uses_single_native_generate(self) -> None:
        """PyTorch prompt batches should use one native generate call when safe."""

        handler = LLMHandler()
        model = _FakeBatchModel()
        tokenizer = _FakeBatchTokenizer()
        handler.llm = model
        handler.llm_tokenizer = tokenizer
        handler.llm_backend = "pt"
        handler.device = "cpu"
        handler.offload_to_cpu = False

        output_texts = handler._run_pt(
            formatted_prompts=["aa", "bbbb"],
            temperature=0.0,
            cfg_scale=1.0,
            negative_prompt="NO USER INPUT",
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            use_constrained_decoding=False,
            generation_phase="understand",
        )

        self.assertEqual(["101|99", "201|202|99"], output_texts)
        self.assertEqual([(2, 4)], model.generate_shapes)
        self.assertEqual("right", tokenizer.padding_side)


if __name__ == "__main__":
    unittest.main()
