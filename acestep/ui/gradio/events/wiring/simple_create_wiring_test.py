"""Tests for simple Create tab event wiring helpers."""

import unittest
from unittest.mock import patch

from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
)
from acestep.ui.gradio.events.wiring.simple_create_wiring import (
    _apply_simple_model_change,
    _apply_simple_tier_change,
    _extract_generation_status,
    _format_enhancement_status,
    _format_simple_status,
)


class SimpleCreateWiringStatusTests(unittest.TestCase):
    """Verify compact simple-tab status formatting."""

    def test_extract_generation_status_formats_backend_slot(self):
        """The advanced status slot should be converted for the simple tab."""

        outputs = [None] * 11
        outputs[10] = "Generation Complete"

        self.assertEqual(
            _extract_generation_status(outputs),
            "Generation complete. Outputs are saved.",
        )

    def test_format_status_compacts_initialization_messages(self):
        """Long init logs should keep the phase and final backend line."""

        status = "Initializing DiT service: acestep-v15-xl-sft\nLoaded on cuda:0"

        self.assertEqual(
            _format_simple_status(status),
            "Loading DiT model...\nLoaded on cuda:0",
        )

    def test_format_status_renames_encoding_phase(self):
        """Encoding messages should read naturally on the simple tab."""

        self.assertEqual(
            _format_simple_status("Encoding & Ready: 1/2"),
            "Encoding audio: 1/2",
        )

    def test_format_enhancement_status_is_compact(self):
        """Enhancement completion should use short simple-tab wording."""

        self.assertEqual(
            _format_enhancement_status("metadata ready", "Caption"),
            "Caption enhanced. Review before generating.\nmetadata ready",
        )

    def test_format_enhancement_status_reports_failure(self):
        """Enhancement failures should be visible in the compact status box."""

        self.assertEqual(
            _format_enhancement_status("5Hz LM not initialized", "Lyrics"),
            "Lyrics enhancement failed.\n5Hz LM not initialized",
        )

    def test_simple_model_selector_applies_turbo_defaults(self):
        """Selecting XL Turbo should update config_path and 8-step model controls."""

        result = _apply_simple_model_change("acestep-v15-xl-turbo", "Custom")

        self.assertEqual(result[0].get("value"), DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(result[1].get("value"), 8)
        self.assertEqual(result[2].get("value"), 1.0)
        self.assertFalse(result[2].get("visible"))
        self.assertTrue(result[9].get("value"))
        self.assertTrue(result[10].get("value"))
        self.assertTrue(result[11].get("value"))
        self.assertTrue(result[15].get("value"))
        self.assertEqual(result[16].get("value"), "double")
        self.assertEqual(result[17].get("value"), 0.02)
        self.assertEqual(result[18].get("value"), 0.06)
        self.assertEqual(result[19].get("value"), "ode")
        self.assertEqual(result[20].get("value"), "euler")
        self.assertEqual(result[21].get("value"), 0.0)
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), "")
        self.assertEqual(result[24].get("value"), "haar")
        self.assertIn("XL Turbo", result[-1])

    def test_simple_model_selector_applies_sft_defaults(self):
        """Selecting XL SFT should update config_path and stable controls."""

        result = _apply_simple_model_change("acestep-v15-xl-sft", "Custom")

        self.assertEqual(result[0].get("value"), DEFAULT_PREMIUM_DIT_MODEL)
        self.assertEqual(result[1].get("value"), 50)
        self.assertEqual(result[2].get("value"), 7.0)
        self.assertTrue(result[2].get("visible"))
        self.assertFalse(result[3].get("value"))
        self.assertEqual(result[4].get("value"), 3.0)
        self.assertEqual(result[4].get("minimum"), 1.0)
        self.assertEqual(result[4].get("maximum"), 5.0)
        self.assertTrue(result[9].get("value"))
        self.assertTrue(result[10].get("value"))
        self.assertFalse(result[11].get("value"))
        self.assertTrue(result[12].get("value"))
        self.assertTrue(result[13].get("value"))
        self.assertTrue(result[14].get("value"))
        self.assertFalse(result[15].get("value"))
        self.assertEqual(result[16].get("value"), "double")
        self.assertEqual(result[17].get("value"), 0.0)
        self.assertEqual(result[18].get("value"), 0.0)
        self.assertEqual(result[19].get("value"), "ode")
        self.assertEqual(result[20].get("value"), "euler")
        self.assertEqual(result[21].get("value"), 0.0)
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), "")
        self.assertEqual(result[24].get("value"), "haar")
        self.assertIn("XL SFT", result[-1])

    def test_simple_model_selector_applies_base_defaults(self):
        """Selecting XL Base should update config_path, base modes, and quality controls."""

        result = _apply_simple_model_change("acestep-v15-xl-base", "Custom")

        self.assertEqual(result[0].get("value"), DEFAULT_BASE_DIT_MODEL)
        self.assertEqual(result[1].get("value"), 64)
        self.assertEqual(result[2].get("value"), 7.0)
        self.assertTrue(result[2].get("visible"))
        self.assertFalse(result[3].get("value"))
        self.assertEqual(result[4].get("value"), 3.0)
        self.assertEqual(result[4].get("minimum"), 1.0)
        self.assertEqual(result[4].get("maximum"), 5.0)
        self.assertIn("Extract", result[8].get("choices"))
        self.assertFalse(result[9].get("value"))
        self.assertFalse(result[10].get("value"))
        self.assertFalse(result[11].get("value"))
        self.assertFalse(result[15].get("value"))
        self.assertEqual(result[16].get("value"), "double")
        self.assertEqual(result[17].get("value"), 0.0)
        self.assertEqual(result[18].get("value"), 0.0)
        self.assertEqual(result[19].get("value"), "ode")
        self.assertEqual(result[20].get("value"), "euler")
        self.assertEqual(result[21].get("value"), 0.0)
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), "")
        self.assertEqual(result[24].get("value"), "haar")
        self.assertIn("XL Base", result[-1])

    def test_simple_tier_selector_mirrors_vram_preset(self):
        """Selecting a VRAM preset should update advanced and simple VRAM controls."""

        tier_updates = (
            {"value": True},  # offload_to_cpu
            {"value": False},  # offload_dit_to_cpu
            {"value": False},  # compile
            {"value": "fp8_scaled"},  # quantization
            {"value": "pt"},  # backend
            {"value": "acestep-5Hz-lm-4B"},  # LM model
            {"value": True},  # init LM
            {"value": 1, "maximum": 1},  # batch
            {"maximum": 480.0},  # duration
            {"value": "gpu info"},  # display
        )

        with patch(
            "acestep.ui.gradio.events.wiring.simple_create_wiring.gen_h.on_tier_change",
            return_value=tier_updates,
        ):
            result = _apply_simple_tier_change("tier6b", llm_handler=object())

        self.assertEqual(result[0].get("value"), "tier6b")
        self.assertEqual(result[1].get("value"), True)
        self.assertEqual(result[4].get("value"), "fp8_scaled")
        self.assertEqual(result[5].get("value"), "fp8_scaled")
        self.assertEqual(result[7].get("value"), "acestep-5Hz-lm-4B")
        self.assertEqual(result[12].get("value"), 1)
        self.assertEqual(result[13].get("maximum"), 480.0)
        self.assertIsNot(result[4], result[5])
        self.assertIsNot(result[9], result[12])
        self.assertIsNot(result[10], result[13])
        self.assertIn("tier6b", result[-1])


if __name__ == "__main__":
    unittest.main()
