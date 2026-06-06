"""Preset round-trip tests for Audio Processing DiffPitcher options."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from acestep.ui.gradio.premium_features import (
    DEFAULT_PRESET_VALUES,
    get_preset_component_keys,
    load_preset_action,
    save_preset_action,
)


class PremiumDiffPitcherPresetTests(unittest.TestCase):
    """Verify custom presets preserve new and existing Audio Processing options."""

    def test_custom_preset_round_trips_diffpitcher_and_existing_ap_options(self) -> None:
        """Saved presets should reload DiffPitcher and existing AP values."""

        with tempfile.TemporaryDirectory() as temp_dir:
            original_root = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = temp_dir
            try:
                reference_path = Path(temp_dir) / "ref.wav"
                midi_path = Path(temp_dir) / "score.mid"
                reference_path.write_bytes(b"placeholder")
                midi_path.write_bytes(b"placeholder")
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
                _set(values, keys, "ap_diffpitcher_enabled", True)
                _set(values, keys, "ap_diffpitcher_mode", "score")
                _set(values, keys, "ap_diffpitcher_reference_audio", str(reference_path))
                _set(values, keys, "ap_diffpitcher_midi", str(midi_path))
                _set(values, keys, "ap_diffpitcher_steps", 80)
                _set(values, keys, "ap_diffpitcher_shift_semitones", -2.5)
                _set(values, keys, "ap_diffpitcher_mask_with_source", False)
                _set(values, keys, "ap_diffpitcher_device", "cpu")
                _set(values, keys, "ap_output_format", "flac")
                _set(values, keys, "ap_lufs", -14.0)

                save_preset_action("diffpitcher-ap", None, *values)
                loaded = load_preset_action("diffpitcher-ap")
            finally:
                if original_root is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original_root

        loaded_values = [_update_value(item) for item in loaded[: len(keys)]]
        self.assertTrue(loaded_values[keys.index("ap_diffpitcher_enabled")])
        self.assertEqual("score", loaded_values[keys.index("ap_diffpitcher_mode")])
        self.assertEqual(
            str(reference_path),
            loaded_values[keys.index("ap_diffpitcher_reference_audio")],
        )
        self.assertEqual(
            str(midi_path),
            loaded_values[keys.index("ap_diffpitcher_midi")],
        )
        self.assertEqual(80, loaded_values[keys.index("ap_diffpitcher_steps")])
        self.assertEqual(
            -2.5,
            loaded_values[keys.index("ap_diffpitcher_shift_semitones")],
        )
        self.assertFalse(loaded_values[keys.index("ap_diffpitcher_mask_with_source")])
        self.assertEqual("cpu", loaded_values[keys.index("ap_diffpitcher_device")])
        self.assertEqual("flac", loaded_values[keys.index("ap_output_format")])
        self.assertEqual(-14.0, loaded_values[keys.index("ap_lufs")])


def _set(values: list[Any], keys: tuple[str, ...], key: str, value: Any) -> None:
    """Set one preset value by component key."""

    values[keys.index(key)] = value


def _update_value(item: Any) -> Any:
    """Extract a Gradio update value when present."""

    return item.get("value") if isinstance(item, dict) and "value" in item else item


if __name__ == "__main__":
    unittest.main()
