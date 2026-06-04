"""Tests for SAM-Audio values in premium custom presets."""

from __future__ import annotations

import os
import tempfile
import unittest

from acestep.ui.gradio import premium_features


class PremiumSamAudioPresetTests(unittest.TestCase):
    """Verify SAM-Audio span settings persist through custom presets."""

    def test_user_preset_saves_and_loads_span_prediction_controls(self) -> None:
        """Predict spans and candidate count should round-trip through presets."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            keys = premium_features.get_preset_component_keys()
            values = [premium_features.DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
            values[keys.index("sam_prompt_mode")] = "text"
            values[keys.index("sam_predict_spans")] = True
            values[keys.index("sam_reranking_candidates")] = 8
            values[keys.index("sam_low_vram_lite")] = False
            try:
                premium_features.save_preset_action("sam span prediction", None, *values)
                loaded = premium_features.load_named_preset("sam span prediction")
                updates = premium_features.load_preset_action("sam span prediction")
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertTrue(loaded["sam_predict_spans"])
        self.assertEqual(8, loaded["sam_reranking_candidates"])
        self.assertTrue(updates[keys.index("sam_predict_spans")].get("value"))
        self.assertEqual(
            8,
            updates[keys.index("sam_reranking_candidates")].get("value"),
        )


if __name__ == "__main__":
    unittest.main()
