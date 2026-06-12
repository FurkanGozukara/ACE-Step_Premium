"""Custom preset tests for the Audio Processing None preset."""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from acestep.audio_processing.presets import PROCESSING_PRESET_NONE, STAGE_KEYS
from acestep.ui.gradio.premium_features import (
    DEFAULT_PRESET_VALUES,
    get_preset_component_keys,
    load_named_preset,
    load_preset_action,
    save_preset_action,
)


class PremiumAudioProcessingNonePresetTests(unittest.TestCase):
    """Verify custom presets preserve the Audio Processing None preset."""

    def test_custom_preset_round_trips_none_processing_preset(self) -> None:
        """Saved presets should reload None and disabled stage checkboxes."""

        with tempfile.TemporaryDirectory() as temp_dir:
            original_root = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = temp_dir
            try:
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
                _set(values, keys, "ap_builtin_preset", PROCESSING_PRESET_NONE)
                for key in STAGE_KEYS:
                    _set(values, keys, f"ap_{key}_enabled", False)

                save_preset_action("ap-none", None, *values)
                loaded_payload = load_named_preset("ap-none")
                loaded_updates = load_preset_action("ap-none")
            finally:
                if original_root is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original_root

        self.assertEqual(PROCESSING_PRESET_NONE, loaded_payload["ap_builtin_preset"])
        for key in STAGE_KEYS:
            self.assertFalse(loaded_payload[f"ap_{key}_enabled"])
            self.assertFalse(_update_value(loaded_updates[keys.index(f"ap_{key}_enabled")]))
        self.assertEqual(
            PROCESSING_PRESET_NONE,
            _update_value(loaded_updates[keys.index("ap_builtin_preset")]),
        )


def _set(values: list[Any], keys: tuple[str, ...], key: str, value: Any) -> None:
    """Set one preset value by component key."""

    values[keys.index(key)] = value


def _update_value(item: Any) -> Any:
    """Extract a Gradio update value when present."""

    return item.get("value") if isinstance(item, dict) and "value" in item else item


if __name__ == "__main__":
    unittest.main()
