"""Tests for batch folder generation orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.core.generation.cancellation import generation_cancel_scope, request_generation_cancel
from acestep.ui.gradio.events.batch_folder_args import (
    BATCH_QUEUE_ARG_INDEX,
    CAPTION_ARG_INDEX,
    GENERATION_ARG_COUNT,
    LYRICS_ARG_INDEX,
)
from acestep.ui.gradio.events.batch_folder_runner import run_batch_folder_processing
from acestep.ui.gradio.events.results.output_manager import (
    create_generation_run_dir,
    write_json,
    write_text,
)
from acestep.ui.gradio.events.results.result_output_contract import (
    ALL_AUDIO_PATHS_INDEX,
    STATUS_INDEX,
)


class BatchFolderRunnerTests(unittest.TestCase):
    """Verify folder items are mapped into generation calls and manifests."""

    def tearDown(self) -> None:
        """Clear cancellation state left by each test."""

        with generation_cancel_scope():
            pass

    def test_runner_overrides_caption_and_lyrics_and_writes_manifest(self):
        """Each lyrics/style pair should produce one generation call."""

        calls = []

        def fake_generation_runner(_dit_handler, _llm_handler, *args):
            calls.append(args)
            run_dir = create_generation_run_dir()
            audio_path = write_text(run_dir / "song.flac", "audio")
            manifest_path = write_json(run_dir / "generation_manifest.json", {"ok": True})
            result = [None] * 55
            result[ALL_AUDIO_PATHS_INDEX] = [audio_path, manifest_path]
            result[STATUS_INDEX] = "Generation Complete"
            yield tuple(result)

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

            self.assertIn("Batch folder processing started", statuses[0])
            self.assertIn("Batch complete: 1/1", statuses[-1])
            self.assertEqual("bright synth pop", calls[0][CAPTION_ARG_INDEX])
            self.assertEqual("[verse]\nhello", calls[0][LYRICS_ARG_INDEX])
            self.assertEqual({}, calls[0][BATCH_QUEUE_ARG_INDEX])

            manifest = json.loads(
                (output_dir / "batch_folder_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("completed", manifest["items"][0]["status"])
            self.assertTrue((output_dir / "song" / "generation_manifest.json").is_file())
            self.assertEqual(
                [
                    str((output_dir / "song" / "song.flac").resolve()).replace("\\", "/"),
                    str((output_dir / "song" / "generation_manifest.json").resolve()).replace(
                        "\\", "/"
                    ),
                ],
                manifest["items"][0]["output_paths"],
            )

    def test_runner_cancel_stops_current_generation_and_remaining_batch(self):
        """A cancel request should stop the active item and skip later files."""

        calls = []

        def cancel_generation_runner(_dit_handler, _llm_handler, *args):
            calls.append(args)
            request_generation_cancel()
            result = [None] * 55
            result[STATUS_INDEX] = "Generation cancelled"
            yield tuple(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            (input_dir / "first.txt").write_text("[verse]\nfirst", encoding="utf-8")
            (input_dir / "second.txt").write_text("[verse]\nsecond", encoding="utf-8")

            base_args = [None] * GENERATION_ARG_COUNT
            base_args[CAPTION_ARG_INDEX] = "style"
            base_args[LYRICS_ARG_INDEX] = "lyrics"
            base_args[BATCH_QUEUE_ARG_INDEX] = {}

            statuses = list(
                run_batch_folder_processing(
                    None,
                    None,
                    str(input_dir),
                    str(output_dir),
                    False,
                    False,
                    base_args,
                    generation_runner=cancel_generation_runner,
                )
            )

        self.assertEqual(1, len(calls))
        self.assertIn("Batch folder processing started", statuses[0])
        self.assertIn("Batch cancelled. Remaining files were not started.", statuses[-1])


if __name__ == "__main__":
    unittest.main()
