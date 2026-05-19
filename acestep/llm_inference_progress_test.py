"""Tests for LLM inference console-progress behavior."""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
