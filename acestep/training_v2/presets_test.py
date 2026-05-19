"""Tests for Training V2 preset defaults."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class TrainingV2PresetDefaultsTests(unittest.TestCase):
    """Verify bundled LoRA presets keep safe shared defaults."""

    def test_lora_presets_keep_alpha_and_disable_dropout(self) -> None:
        """Every bundled preset should use alpha 128 and dropout 0."""

        preset_dir = Path(__file__).resolve().parent / "presets"
        for preset_path in preset_dir.glob("*.json"):
            with self.subTest(preset=preset_path.name):
                preset = json.loads(preset_path.read_text(encoding="utf-8"))
                self.assertEqual(128, preset["alpha"])
                self.assertEqual(0.0, preset["dropout"])


if __name__ == "__main__":
    unittest.main()
