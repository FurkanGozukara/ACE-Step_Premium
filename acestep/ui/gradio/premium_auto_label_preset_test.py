"""Tests for dataset auto-label controls in premium user presets."""

from __future__ import annotations

import os
import tempfile
import unittest

from acestep.ui.gradio import premium_features


class PremiumAutoLabelPresetTests(unittest.TestCase):
    """Verify auto-label options persist through the shared preset system."""

    def _set_project_root(self, tmp_dir: str) -> str | None:
        """Set the test project root and return the previous value."""

        original = os.environ.get("ACESTEP_PROJECT_ROOT")
        os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
        return original

    def _restore_project_root(self, original: str | None) -> None:
        """Restore the project root environment variable."""

        if original is None:
            os.environ.pop("ACESTEP_PROJECT_ROOT", None)
        else:
            os.environ["ACESTEP_PROJECT_ROOT"] = original

    def test_user_preset_saves_and_loads_auto_label_options(self) -> None:
        """Auto-label batch size and sibling options should round-trip."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("skip_metas")] = True
            values[keys.index("only_unlabeled")] = False
            values[keys.index("auto_label_output_dir")] = "datasets/labels"
            values[keys.index("auto_label_subprocess")] = False
            values[keys.index("auto_label_batch_size")] = 8
            values[keys.index("transcribe_lyrics")] = True
            values[keys.index("lm_lyrics_language")] = "en"
            values[keys.index("use_only_custom_trigger")] = True
            try:
                premium_features.save_preset_action("auto label", None, *values)
                loaded = premium_features.load_named_preset("auto label")
                updates = premium_features.load_preset_action("auto label")
            finally:
                self._restore_project_root(original)

        self.assertEqual(8, loaded["auto_label_batch_size"])
        self.assertFalse(loaded["auto_label_subprocess"])
        self.assertTrue(loaded["skip_metas"])
        self.assertFalse(loaded["only_unlabeled"])
        self.assertTrue(loaded["use_only_custom_trigger"])
        self.assertEqual(
            8,
            updates[keys.index("auto_label_batch_size")].get("value"),
        )
        self.assertFalse(updates[keys.index("auto_label_subprocess")].get("value"))
        self.assertTrue(updates[keys.index("use_only_custom_trigger")].get("value"))

    def test_user_preset_saves_and_loads_lm_audio_code_toggle(self) -> None:
        """The LM audio-code checkbox should round-trip through user presets."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            toggle_index = keys.index("generate_lm_audio_codes")
            values[toggle_index] = False
            try:
                premium_features.save_preset_action("lm codes off", None, *values)
                loaded_off = premium_features.load_named_preset("lm codes off")
                updates_off = premium_features.load_preset_action("lm codes off")

                values[toggle_index] = True
                premium_features.save_preset_action("lm codes on", None, *values)
                loaded_on = premium_features.load_named_preset("lm codes on")
                updates_on = premium_features.load_preset_action("lm codes on")
            finally:
                self._restore_project_root(original)

        self.assertFalse(loaded_off["generate_lm_audio_codes"])
        self.assertFalse(updates_off[toggle_index].get("value"))
        self.assertTrue(loaded_on["generate_lm_audio_codes"])
        self.assertTrue(updates_on[toggle_index].get("value"))


if __name__ == "__main__":
    unittest.main()
