"""Tests for preprocessing debug text prompt files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.training.dataset_builder_modules.preprocess_debug_text import (
    DEBUG_TEXT_PROMPT_DIRNAME,
    save_debug_text_prompt,
)
from acestep.training.dataset_builder_modules.models import AudioSample
from acestep.training.dataset_builder_modules.preprocess_text import build_text_prompt


class PreprocessDebugTextTests(unittest.TestCase):
    """Verify exact text prompts are written for debugging."""

    def test_save_debug_text_prompt_writes_exact_inputs(self) -> None:
        """Prompt debug output should preserve exact text and lyric inputs."""

        prompt = "ohwx, west coast hip hop\n- bpm: 92\n"
        lyrics = "[Verse]\nwords"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_debug_text_prompt(tmpdir, "sample:01", prompt, lyrics)

            debug_path = Path(path)
            self.assertEqual(DEBUG_TEXT_PROMPT_DIRNAME, debug_path.parent.name)
            self.assertEqual("sample_01.txt", debug_path.name)
            self.assertEqual(
                f"# Text Encoder Input\n{prompt}\n\n# Lyrics Encoder Input\n{lyrics}",
                debug_path.read_text(encoding="utf-8"),
            )

    def test_build_text_prompt_contains_prepended_custom_tag(self) -> None:
        """The final text prompt should include the custom trigger tag."""

        sample = AudioSample(
            caption="bright pop",
            custom_tag="ohwx",
            bpm=120,
            keyscale="C major",
            timesignature="4",
            duration=30,
        )

        prompt = build_text_prompt(sample, "prepend", use_genre=False)

        self.assertIn("ohwx, bright pop", prompt)
        self.assertIn("- bpm: 120", prompt)


if __name__ == "__main__":
    unittest.main()
