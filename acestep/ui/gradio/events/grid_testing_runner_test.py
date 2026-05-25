"""Tests for the Grid Testing runner."""

from __future__ import annotations

from contextvars import Context
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.grid_testing_args import (
    AUDIO_FORMAT_ARG_INDEX,
    BATCH_SIZE_ARG_INDEX,
    LORA_DROPDOWN_ARG_INDEX,
    RANDOM_SEED_ARG_INDEX,
    SEED_ARG_INDEX,
    USE_LORA_ARG_INDEX,
)
from acestep.ui.gradio.events.grid_testing_runner import run_grid_testing
from acestep.ui.gradio.events.results.output_manager import create_generation_run_dir


class GridTestingRunnerTests(unittest.TestCase):
    """Verify Grid Testing orchestrates generation args and flattened output."""

    def test_random_seed_is_fixed_for_all_lora_jobs(self) -> None:
        """Random seed mode should become one fixed comparable grid seed."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "grid"
            lora_path = Path(tmpdir) / "voice.safetensors"
            lora_path.write_bytes(b"placeholder")
            captured_args: list[tuple[bool, str, str, bool, str, int]] = []

            def fake_runner(_dit_handler, _llm_handler, *args):
                captured_args.append(
                    (
                        bool(args[RANDOM_SEED_ARG_INDEX]),
                        str(args[SEED_ARG_INDEX]),
                        str(args[LORA_DROPDOWN_ARG_INDEX] or ""),
                        bool(args[USE_LORA_ARG_INDEX]),
                        str(args[AUDIO_FORMAT_ARG_INDEX]),
                        int(args[BATCH_SIZE_ARG_INDEX]),
                    )
                )
                paths = _write_fake_generated_run("0001")
                result = [None] * 55
                result[8] = paths
                result[10] = "Generation Complete"
                yield tuple(result)

            with patch(
                "acestep.ui.gradio.events.grid_testing_args.random.randint",
                return_value=12345,
            ):
                outputs = list(
                    run_grid_testing(
                        object(),
                        object(),
                        ["", str(lora_path)],
                        str(output_dir),
                        True,
                        _generation_args(),
                        generations_per_lora=3,
                        generation_runner=fake_runner,
                    )
                )

            final_status, final_files = outputs[-1]
            final_paths = final_files["value"]

        self.assertIn("Grid complete", final_status)
        self.assertEqual(
            [
                (False, "12345", "", False, "mp3", 3),
                (False, "12345", str(lora_path.resolve()), True, "mp3", 3),
            ],
            captured_args,
        )
        self.assertIn("Grid jobs: 2 LoRA(s), 6 song(s) total.", final_status)
        self.assertIn("base-model-0001.mp3", " ".join(final_paths))
        self.assertIn("voice-0001.mp3", " ".join(final_paths))
        self.assertFalse((output_dir / ".grid_work").exists())

    def test_streamed_grid_output_scope_survives_context_switches(self) -> None:
        """Grid output overrides should survive Gradio streaming context changes."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "grid"

            def fake_runner(_dit_handler, _llm_handler, *_args):
                pending = [None] * 55
                pending[10] = "Preparing generation"
                yield tuple(pending)

                paths = _write_fake_generated_run("0001")
                final = [None] * 55
                final[8] = paths
                final[10] = "Generation Complete"
                yield tuple(final)

            iterator = run_grid_testing(
                object(),
                object(),
                [""],
                str(output_dir),
                True,
                _generation_args(),
                generation_runner=fake_runner,
            )
            outputs = _drain_in_fresh_contexts(iterator)

            final_status, final_files = outputs[-1]
            final_paths = final_files["value"]

        self.assertIn("Grid complete", final_status)
        self.assertIn("base-model-0001.mp3", " ".join(final_paths))
        self.assertFalse((output_dir / ".grid_work").exists())


def _generation_args() -> list[object]:
    """Return a minimal generation argument list matching the UI contract."""

    args: list[object] = [None] * 94
    args[0] = "style"
    args[1] = "lyrics"
    args[12] = 1
    args[RANDOM_SEED_ARG_INDEX] = True
    args[SEED_ARG_INDEX] = "-1"
    args[AUDIO_FORMAT_ARG_INDEX] = "flac_mp3"
    return args


def _drain_in_fresh_contexts(iterator) -> list[tuple[object, object]]:
    """Drain an iterator while simulating Gradio context switches."""

    outputs = []
    while True:
        try:
            outputs.append(Context().run(next, iterator))
        except StopIteration:
            return outputs


def _write_fake_generated_run(key: str) -> list[str]:
    """Write fake generated artifacts in the active output run folder."""

    run_dir = create_generation_run_dir()
    mp3_path = run_dir / f"{key}.mp3"
    metadata_path = run_dir / f"{key}.json"
    manifest_path = run_dir / "generation_manifest.json"
    mp3_path.write_text("audio", encoding="utf-8")
    metadata_path.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_index": 1,
                        "audio_paths": {"mp3": str(mp3_path)},
                        "mp3_path": str(mp3_path),
                        "metadata_path": str(metadata_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return [str(mp3_path), str(metadata_path), str(manifest_path)]


if __name__ == "__main__":
    unittest.main()
