"""Tests for Grid Testing LoRA selection helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.grid_testing_loras import (
    BASE_MODEL_PREFIX,
    filter_grid_lora_choices,
    resolve_grid_lora_jobs,
)


class GridTestingLoraTests(unittest.TestCase):
    """Verify Grid Testing LoRA selections become deterministic jobs."""

    def test_empty_selection_defaults_to_base_model(self) -> None:
        """No selected LoRA should still generate the base model comparison."""

        jobs = resolve_grid_lora_jobs([])

        self.assertEqual(1, len(jobs))
        self.assertEqual("", jobs[0].path)
        self.assertEqual(BASE_MODEL_PREFIX, jobs[0].prefix)

    def test_safetensors_selection_uses_file_stem_prefix(self) -> None:
        """A selected safetensors file should use its filename stem as prefix."""

        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "2pac-sft-1e-04-epoch-25.safetensors"
            lora_path.write_bytes(b"placeholder")

            jobs = resolve_grid_lora_jobs(["", str(lora_path)])

        self.assertEqual(["base-model", "2pac-sft-1e-04-epoch-25"], [job.prefix for job in jobs])

    def test_duplicate_prefixes_get_unique_suffixes(self) -> None:
        """Duplicate LoRA basenames should not overwrite each other in one grid."""

        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "a" / "voice.safetensors"
            second = Path(tmpdir) / "b" / "voice.safetensors"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"placeholder")
            second.write_bytes(b"placeholder")

            jobs = resolve_grid_lora_jobs([str(first), str(second)])

        self.assertEqual(["voice", "voice-2"], [job.prefix for job in jobs])

    def test_filter_choices_matches_label_and_path(self) -> None:
        """LoRA filter should match both visible labels and filesystem paths."""

        choices = [
            ("None", ""),
            ("Warm Vocal", "C:/Loras/warm-vocal.safetensors"),
            ("Clean Guitar", "C:/Loras/guitar.safetensors"),
        ]

        visible_choices, selected = filter_grid_lora_choices(
            choices,
            filter_text="guitar",
            selected_loras=[""],
        )

        self.assertEqual(
            ["", "C:/Loras/guitar.safetensors"],
            [value for _label, value in visible_choices],
        )
        self.assertEqual([""], selected)

    def test_filter_choices_preserves_selected_hidden_loras(self) -> None:
        """Selected LoRAs should remain available when the filter changes."""

        choices = [
            ("None", ""),
            ("Warm Vocal", "C:/Loras/warm-vocal.safetensors"),
            ("Clean Guitar", "C:/Loras/guitar.safetensors"),
        ]

        visible_choices, selected = filter_grid_lora_choices(
            choices,
            filter_text="guitar",
            selected_loras=["C:/Loras/warm-vocal.safetensors"],
        )

        self.assertEqual(
            [
                "C:/Loras/warm-vocal.safetensors",
                "C:/Loras/guitar.safetensors",
            ],
            [value for _label, value in visible_choices],
        )
        self.assertEqual(["C:/Loras/warm-vocal.safetensors"], selected)


if __name__ == "__main__":
    unittest.main()
