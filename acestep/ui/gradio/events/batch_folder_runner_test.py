"""Tests for batch folder generation orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.batch_folder_args import (
    BATCH_QUEUE_ARG_INDEX,
    CAPTION_ARG_INDEX,
    GENERATION_ARG_COUNT,
    LYRICS_ARG_INDEX,
)
from acestep.ui.gradio.events.batch_folder_runner import run_batch_folder_processing


class BatchFolderRunnerTests(unittest.TestCase):
    """Verify folder items are mapped into generation calls and manifests."""

    def test_runner_overrides_caption_and_lyrics_and_writes_manifest(self):
        """Each lyrics/style pair should produce one generation call."""

        calls = []

        def fake_generation_runner(_dit_handler, _llm_handler, *args):
            calls.append(args)
            yield (None,) * 8 + (["target/song.flac"], "info", "Generation Complete")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            (input_dir / "song.txt").write_text("[verse]\nhello", encoding="utf-8")
            (input_dir / "song_style.txt").write_text("bright synth pop", encoding="utf-8")

            base_args = [None] * GENERATION_ARG_COUNT
            base_args[CAPTION_ARG_INDEX] = "fallback style"
            base_args[LYRICS_ARG_INDEX] = "fallback lyrics"
            base_args[BATCH_QUEUE_ARG_INDEX] = ["stale"]

            statuses = list(
                run_batch_folder_processing(
                    None,
                    None,
                    str(input_dir),
                    str(output_dir),
                    False,
                    False,
                    base_args,
                    generation_runner=fake_generation_runner,
                )
            )

            self.assertIn("Batch complete: 1/1", statuses[-1])
            self.assertEqual("bright synth pop", calls[0][CAPTION_ARG_INDEX])
            self.assertEqual("[verse]\nhello", calls[0][LYRICS_ARG_INDEX])
            self.assertEqual({}, calls[0][BATCH_QUEUE_ARG_INDEX])

            manifest = json.loads(
                (output_dir / "batch_folder_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("completed", manifest["items"][0]["status"])
            self.assertEqual(["target/song.flac"], manifest["items"][0]["output_paths"])


if __name__ == "__main__":
    unittest.main()
