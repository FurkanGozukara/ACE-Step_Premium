"""Tests for LoRA browser UI status helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.generation import lora_browser


class LoraBrowserTests(unittest.TestCase):
    """Verify the visible LoRA status mirrors next-generation behavior."""

    def test_none_selection_disables_lora(self) -> None:
        """Selecting None should clear manual path and mark LoRA disabled."""

        path_update, status, use_update = lora_browser.select_lora_dropdown_path("")

        self.assertEqual(path_update.get("value"), "")
        self.assertEqual(status, "No LoRA will be used.")
        self.assertFalse(use_update.get("value"))

    def test_valid_manual_path_wins_over_dropdown(self) -> None:
        """Manual input should be the effective path when both fields are set."""

        with patch.object(
            lora_browser,
            "resolve_loadable_lora_adapter_path",
            return_value=r"G:\ACE_Step_v1\Loras\manual",
        ) as resolver:
            status, use_update = lora_browser.update_lora_next_run_status(
                r".\Loras\manual",
                r".\Loras\dropdown",
            )

        resolver.assert_called_once_with(r".\Loras\manual")
        self.assertIn("Next run will use LoRA:", status)
        self.assertTrue(use_update.get("value"))

    def test_invalid_path_reports_base_model(self) -> None:
        """Invalid LoRA paths should explicitly say the base model will be used."""

        with patch.object(
            lora_browser,
            "resolve_loadable_lora_adapter_path",
            return_value="",
        ):
            status, use_update = lora_browser.update_lora_next_run_status("missing", "")

        self.assertEqual(status, "No LoRA will be used. Invalid LoRA path: missing")
        self.assertFalse(use_update.get("value"))


if __name__ == "__main__":
    unittest.main()
