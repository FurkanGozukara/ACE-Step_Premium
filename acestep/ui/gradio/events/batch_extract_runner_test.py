"""Tests for Advanced-tab batch processing."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any

from acestep.core.generation.cancellation import (
    generation_cancel_scope,
    request_generation_cancel,
)
from acestep.ui.gradio.events.batch_extract_runner import (
    BATCH_SIZE_ARG_INDEX,
    EXTRACT_ALL_STEMS_ARG_INDEX,
    SRC_AUDIO_ARG_INDEX,
    TASK_TYPE_ARG_INDEX,
    TRACK_NAME_ARG_INDEX,
    run_batch_extract_processing,
)
from acestep.ui.gradio.events.batch_folder_args import (
    AUDIO_DURATION_ARG_INDEX,
    AUTOGEN_ARG_INDEX,
    BATCH_QUEUE_ARG_INDEX,
    CURRENT_BATCH_INDEX_ARG_INDEX,
    GENERATION_PARAMS_STATE_ARG_INDEX,
    TOTAL_BATCHES_ARG_INDEX,
)
from acestep.ui.gradio.events.results.result_output_contract import (
    ALL_AUDIO_PATHS_INDEX,
    STATUS_INDEX,
)


def _write_wav(path: Path, duration_seconds: float = 0.25) -> None:
    """Write a small valid PCM WAV file for duration and folder tests."""

    sample_rate = 8_000
    frames = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def _generation_args() -> list[Any]:
    """Return generation args with the fields batch processing mutates populated."""

    args: list[Any] = [None] * 100
    args[AUDIO_DURATION_ARG_INDEX] = 999
    args[BATCH_SIZE_ARG_INDEX] = 8
    args[SRC_AUDIO_ARG_INDEX] = "stale.wav"
    args[TASK_TYPE_ARG_INDEX] = "extract"
    args[TRACK_NAME_ARG_INDEX] = "vocals"
    args[EXTRACT_ALL_STEMS_ARG_INDEX] = False
    args[AUTOGEN_ARG_INDEX] = True
    args[CURRENT_BATCH_INDEX_ARG_INDEX] = 5
    args[TOTAL_BATCHES_ARG_INDEX] = 9
    args[BATCH_QUEUE_ARG_INDEX] = {"stale": True}
    args[GENERATION_PARAMS_STATE_ARG_INDEX] = {"stale": True}
    return args


def _result(paths: list[str], status: str = "Generation Complete") -> tuple[Any, ...]:
    """Return a generation-result tuple with path and status slots populated."""

    values: list[Any] = [None] * 55
    values[ALL_AUDIO_PATHS_INDEX] = paths
    values[STATUS_INDEX] = status
    return tuple(values)


class BatchExtractRunnerTests(unittest.TestCase):
    """Verify folder Extract sequencing, output copying, and cancellation."""

    def tearDown(self) -> None:
        """Clear cancellation state left by cancellation-focused tests."""

        with generation_cancel_scope():
            pass

    def test_processes_audio_files_and_copies_outputs_with_source_names(self) -> None:
        """Batch processing uses real audio files and saves outputs with input stems."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            generated_dir = root / "generated"
            input_dir.mkdir()
            generated_dir.mkdir()
            _write_wav(input_dir / "Alpha.wav", duration_seconds=0.5)
            _write_wav(input_dir / "Beta.wav", duration_seconds=1.0)
            (input_dir / "ignore.txt").write_text("not audio", encoding="utf-8")
            calls: list[tuple[Any, ...]] = []

            def fake_runner(_dit, _llm, *args):
                """Create real generated files and return them like the pipeline."""

                calls.append(args)
                source = Path(args[SRC_AUDIO_ARG_INDEX])
                flac_path = generated_dir / f"generated-{source.stem}.flac"
                remaining_path = generated_dir / f"generated-{source.stem}_remaining.flac"
                mp3_path = generated_dir / f"generated-{source.stem}.mp3"
                flac_path.write_bytes(b"flac-data")
                remaining_path.write_bytes(b"remaining-data")
                mp3_path.write_bytes(b"mp3-data")
                yield _result(
                    [
                        str(flac_path),
                        str(remaining_path),
                        str(mp3_path),
                        str(flac_path.with_suffix(".json")),
                    ]
                )

            statuses = list(
                run_batch_extract_processing(
                    None,
                    None,
                    str(input_dir),
                    str(output_dir),
                    _generation_args(),
                    generation_runner=fake_runner,
                )
            )

            self.assertIn("Batch Process complete: 2/2 file(s) saved", statuses[-1])
            self.assertEqual(b"flac-data", (output_dir / "Alpha.flac").read_bytes())
            self.assertEqual(
                b"remaining-data",
                (output_dir / "Alpha_remaining.flac").read_bytes(),
            )
            self.assertEqual(b"mp3-data", (output_dir / "Alpha.mp3").read_bytes())
            self.assertEqual(b"flac-data", (output_dir / "Beta.flac").read_bytes())
            self.assertEqual(
                b"remaining-data",
                (output_dir / "Beta_remaining.flac").read_bytes(),
            )
            self.assertEqual(2, len(calls))
            self.assertEqual("extract", calls[0][TASK_TYPE_ARG_INDEX])
            self.assertEqual("vocals", calls[0][TRACK_NAME_ARG_INDEX])
            self.assertEqual(1, calls[0][BATCH_SIZE_ARG_INDEX])
            self.assertFalse(calls[0][AUTOGEN_ARG_INDEX])
            self.assertEqual({}, calls[0][BATCH_QUEUE_ARG_INDEX])
            self.assertAlmostEqual(0.5, calls[0][AUDIO_DURATION_ARG_INDEX], places=2)
            self.assertAlmostEqual(1.0, calls[1][AUDIO_DURATION_ARG_INDEX], places=2)

    def test_extract_all_stems_processes_every_stem_with_suffixes(self) -> None:
        """Batch Process can extract every stem for each source audio file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            generated_dir = root / "generated"
            input_dir.mkdir()
            generated_dir.mkdir()
            _write_wav(input_dir / "Alpha.wav")
            args = _generation_args()
            args[TRACK_NAME_ARG_INDEX] = None
            args[EXTRACT_ALL_STEMS_ARG_INDEX] = True
            calls: list[tuple[Any, ...]] = []

            def fake_runner(_dit, _llm, *run_args):
                """Create one generated file for the requested stem."""

                calls.append(run_args)
                track_name = run_args[TRACK_NAME_ARG_INDEX]
                path = generated_dir / f"{track_name}.flac"
                path.write_bytes(str(track_name).encode("utf-8"))
                yield _result([str(path)])

            statuses = list(
                run_batch_extract_processing(
                    None,
                    None,
                    str(input_dir),
                    str(output_dir),
                    args,
                    generation_runner=fake_runner,
                )
            )

            called_tracks = [call[TRACK_NAME_ARG_INDEX] for call in calls]
            self.assertEqual(
                [
                    "woodwinds",
                    "brass",
                    "fx",
                    "synth",
                    "strings",
                    "percussion",
                    "keyboard",
                    "guitar",
                    "bass",
                    "drums",
                    "backing_vocals",
                    "vocals",
                ],
                called_tracks,
            )
            self.assertFalse(any(call[EXTRACT_ALL_STEMS_ARG_INDEX] for call in calls))
            self.assertTrue((output_dir / "Alpha_woodwinds.flac").is_file())
            self.assertTrue((output_dir / "Alpha_guitar.flac").is_file())
            self.assertTrue((output_dir / "Alpha_vocal.flac").is_file())
            for call, expected_stem in zip(calls, called_tracks):
                self.assertEqual(expected_stem, call[0])
                self.assertEqual("", call[1])
                self.assertEqual(expected_stem, call[TRACK_NAME_ARG_INDEX])
                self.assertEqual(
                    f"Extract the {expected_stem.upper()} track from the audio:",
                    call[19],
                )
                self.assertFalse(call[EXTRACT_ALL_STEMS_ARG_INDEX])
            self.assertIn("Batch Process complete: 1/1 file(s) saved", statuses[-1])

    def test_output_folder_is_mandatory(self) -> None:
        """A missing output folder stops before any generation starts."""

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            _write_wav(input_dir / "Alpha.wav")
            calls: list[tuple[Any, ...]] = []

            statuses = list(
                run_batch_extract_processing(
                    None,
                    None,
                    str(input_dir),
                    "",
                    _generation_args(),
                    generation_runner=lambda *_args: calls.append(_args),
                )
            )

            self.assertEqual(["Enter a Batch Process Output Folder before starting."], statuses)
            self.assertEqual([], calls)

    def test_cancel_stops_remaining_files(self) -> None:
        """A user cancel request aborts the folder after the active item observes it."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            _write_wav(input_dir / "Alpha.wav")
            _write_wav(input_dir / "Beta.wav")
            calls = 0

            def cancelling_runner(_dit, _llm, *args):
                """Request cancellation while the first item is streaming."""

                nonlocal calls
                calls += 1
                request_generation_cancel()
                yield _result([], "cancelled")

            statuses = list(
                run_batch_extract_processing(
                    None,
                    None,
                    str(input_dir),
                    str(output_dir),
                    _generation_args(),
                    generation_runner=cancelling_runner,
                )
            )

            self.assertEqual(1, calls)
            self.assertIn("Generation cancelled by user.", statuses[-1])
            self.assertIn("Remaining files were not started.", statuses[-1])
            self.assertFalse(any(output_dir.glob("*.*")))


if __name__ == "__main__":
    unittest.main()
