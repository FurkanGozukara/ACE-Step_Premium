"""Tests for sampler mode values in premium user presets."""

from __future__ import annotations

import os
import tempfile
import unittest

from acestep.ui.gradio import premium_features


class PremiumSamplerModePresetTests(unittest.TestCase):
    """Verify Generate Song and advanced sampler modes round-trip in presets."""

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

    def test_user_preset_saves_generate_song_sampler_mode(self) -> None:
        """A sampler chosen on Generate Song should load into both synced fields."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("sampler_mode")] = "heun"
            values[keys.index("simple_create_sampler_mode")] = "euler"
            try:
                premium_features.save_preset_action("sampler mode", None, *values)
                loaded = premium_features.load_named_preset("sampler mode")
                updates = premium_features.load_preset_action("sampler mode")
            finally:
                self._restore_project_root(original)

        self.assertEqual("euler", loaded["sampler_mode"])
        self.assertEqual("euler", loaded["simple_create_sampler_mode"])
        self.assertEqual("euler", updates[keys.index("sampler_mode")].get("value"))
        self.assertEqual(
            "euler",
            updates[keys.index("simple_create_sampler_mode")].get("value"),
        )

    def test_missing_sampler_values_default_to_heun(self) -> None:
        """Older presets without sampler fields should default both controls to Heun."""

        payload = premium_features._apply_runtime_defaults({})

        self.assertEqual("heun", payload["sampler_mode"])
        self.assertEqual("heun", payload["simple_create_sampler_mode"])


if __name__ == "__main__":
    unittest.main()
