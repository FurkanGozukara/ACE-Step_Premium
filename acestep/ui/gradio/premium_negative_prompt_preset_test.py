"""Tests for negative prompt values in premium user presets."""

from __future__ import annotations

import os
import tempfile
import unittest

from acestep.ui.gradio import premium_features


class PremiumNegativePromptPresetTests(unittest.TestCase):
    """Verify Generate Song and advanced negative prompts round-trip in presets."""

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

    def test_user_preset_saves_generate_song_negative_prompt(self) -> None:
        """A value typed on Generate Song should load into both synced fields."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("simple_create_negative_prompt")] = "clipped noise"
            try:
                premium_features.save_preset_action("negative prompt", None, *values)
                loaded = premium_features.load_named_preset("negative prompt")
                updates = premium_features.load_preset_action("negative prompt")
            finally:
                self._restore_project_root(original)

        self.assertEqual("clipped noise", loaded["lm_negative_prompt"])
        self.assertEqual("clipped noise", loaded["simple_create_negative_prompt"])
        self.assertEqual(
            "clipped noise",
            updates[keys.index("lm_negative_prompt")].get("value"),
        )
        self.assertEqual(
            "clipped noise",
            updates[keys.index("simple_create_negative_prompt")].get("value"),
        )

    def test_user_preset_backfills_generate_song_negative_prompt(self) -> None:
        """Older advanced-only preset values should still populate Generate Song."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("lm_negative_prompt")] = "harsh resonance"
            try:
                premium_features.save_preset_action("advanced negative", None, *values)
                loaded = premium_features.load_named_preset("advanced negative")
                updates = premium_features.load_preset_action("advanced negative")
            finally:
                self._restore_project_root(original)

        self.assertEqual("harsh resonance", loaded["lm_negative_prompt"])
        self.assertEqual("harsh resonance", loaded["simple_create_negative_prompt"])
        self.assertEqual(
            "harsh resonance",
            updates[keys.index("simple_create_negative_prompt")].get("value"),
        )

    def test_user_preset_negative_prompt_sentinel_loads_blank(self) -> None:
        """The old NO USER INPUT sentinel should not reappear from presets."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("lm_negative_prompt")] = "NO USER INPUT"
            try:
                premium_features.save_preset_action("sentinel negative", None, *values)
                loaded = premium_features.load_named_preset("sentinel negative")
                updates = premium_features.load_preset_action("sentinel negative")
            finally:
                self._restore_project_root(original)

        self.assertEqual("", loaded["lm_negative_prompt"])
        self.assertEqual("", loaded["simple_create_negative_prompt"])
        self.assertEqual("", updates[keys.index("lm_negative_prompt")].get("value"))
        self.assertEqual(
            "",
            updates[keys.index("simple_create_negative_prompt")].get("value"),
        )


if __name__ == "__main__":
    unittest.main()
