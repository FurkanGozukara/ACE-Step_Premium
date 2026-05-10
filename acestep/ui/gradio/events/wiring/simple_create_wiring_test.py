"""Tests for simple Create tab event wiring helpers."""

import unittest

from acestep.ui.gradio.events.wiring.simple_create_wiring import (
    _apply_simple_model_change,
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

        self.assertEqual(result[0].get("value"), "acestep-v15-xl-turbo")
        self.assertEqual(result[1].get("value"), 8)
        self.assertEqual(result[2].get("value"), 1.0)
        self.assertFalse(result[2].get("visible"))
        self.assertIn("XL Turbo", result[-1])

    def test_simple_model_selector_applies_sft_defaults(self):
        """Selecting XL SFT should update config_path and 50-step model controls."""

        result = _apply_simple_model_change("acestep-v15-xl-sft", "Custom")

        self.assertEqual(result[0].get("value"), "acestep-v15-xl-sft")
        self.assertEqual(result[1].get("value"), 50)
        self.assertEqual(result[2].get("value"), 7.0)
        self.assertTrue(result[2].get("visible"))
        self.assertIn("XL SFT", result[-1])

    def test_simple_model_selector_applies_base_defaults(self):
        """Selecting XL Base should update config_path, base modes, and 50-step controls."""

        result = _apply_simple_model_change("acestep-v15-xl-base", "Custom")

        self.assertEqual(result[0].get("value"), "acestep-v15-xl-base")
        self.assertEqual(result[1].get("value"), 50)
        self.assertEqual(result[2].get("value"), 7.0)
        self.assertTrue(result[2].get("visible"))
        self.assertIn("Extract", result[8].get("choices"))
        self.assertIn("XL Base", result[-1])


if __name__ == "__main__":
    unittest.main()
