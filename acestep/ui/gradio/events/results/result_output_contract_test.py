"""Tests for shared generation result output contracts."""

import importlib.util
import unittest
from pathlib import Path


def _load_contract_module():
    """Load the contract module without importing the full Gradio app."""

    module_path = Path(__file__).resolve().with_name("result_output_contract.py")
    spec = importlib.util.spec_from_file_location("result_output_contract_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_CONTRACT = _load_contract_module()
is_bounded_source_edit = _CONTRACT.is_bounded_source_edit
source_audio_update_path = _CONTRACT.source_audio_update_path
extract_latest_edit_area_paths = _CONTRACT.extract_latest_edit_area_paths
extract_lego_generated_track_path = _CONTRACT.extract_lego_generated_track_path


class ResultOutputContractTests(unittest.TestCase):
    """Validate source-audio comparison and bounded-edit helpers."""

    def test_repaint_lego_and_complete_are_bounded_when_range_is_valid(self):
        """Bounded source edits require a supported task and end greater than start."""

        for task_type in ("repaint", "lego", "complete"):
            self.assertTrue(is_bounded_source_edit(task_type, 10, 20))

    def test_full_source_tasks_are_not_bounded_edits(self):
        """Cover/remix-style tasks can compare source audio without range preservation."""

        self.assertFalse(is_bounded_source_edit("cover", 10, 20))
        self.assertFalse(is_bounded_source_edit("repaint", 10, 10))
        self.assertFalse(is_bounded_source_edit("repaint", 20, 10))
        self.assertFalse(is_bounded_source_edit("text2music", 10, 20))

    def test_source_audio_path_is_only_visible_for_compare_tasks(self):
        """Only source-based tasks expose an original input player."""

        self.assertEqual(
            source_audio_update_path("repaint", r"C:\input\song.wav"),
            "C:/input/song.wav",
        )
        self.assertIsNone(source_audio_update_path("text2music", r"C:\input\song.wav"))

    def test_extract_latest_edit_area_paths_finds_generated_and_original(self):
        """Inline preview should locate edited-area clips by stable filenames."""

        generated, original = extract_latest_edit_area_paths(
            [
                r"C:\run\sample.flac",
                r"C:\run\sample_latest_repainted_area.wav",
                r"C:\run\sample_latest_repainted_area_original.wav",
            ]
        )

        self.assertEqual(generated, "C:/run/sample_latest_repainted_area.wav")
        self.assertEqual(original, "C:/run/sample_latest_repainted_area_original.wav")

    def test_extract_latest_edit_area_paths_ignores_full_lego_generated_track(self):
        """The full raw Lego layer should not be shown as the edited-area clip."""

        generated, original = extract_latest_edit_area_paths(
            [
                r"C:\run\sample.flac",
                r"C:\run\sample_lego_generated_track.wav",
            ]
        )

        self.assertIsNone(generated)
        self.assertIsNone(original)

    def test_extract_lego_generated_track_path_finds_raw_lego_layer(self):
        """The raw Lego layer should be exposed through its own output row."""

        path = extract_lego_generated_track_path(
            [
                r"C:\run\sample.flac",
                r"C:\run\sample_lego_generated_track.wav",
            ]
        )

        self.assertEqual(path, "C:/run/sample_lego_generated_track.wav")


if __name__ == "__main__":
    unittest.main()
