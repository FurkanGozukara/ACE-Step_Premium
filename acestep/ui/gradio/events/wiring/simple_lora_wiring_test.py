"""Tests for simple-tab LoRA synchronization helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring import simple_lora_wiring


class SimpleLoraWiringTests(unittest.TestCase):
    """Verify simple LoRA controls mirror into the advanced generation path."""

    def test_simple_lora_dropdown_updates_advanced_controls(self) -> None:
        """Selecting a LoRA in the simple tab should set the advanced LoRA fields."""

        with patch.object(
            simple_lora_wiring.gen_h,
            "select_lora_dropdown_path",
            return_value=({"value": "adapter"}, "Next run will use LoRA: adapter", {"value": True}),
        ):
            dropdown_update, path_update, status, use_update = (
                simple_lora_wiring.sync_simple_lora_dropdown("adapter")
            )

        self.assertEqual("adapter", dropdown_update.get("value"))
        self.assertEqual("adapter", path_update.get("value"))
        self.assertIn("Next run will use LoRA", status)
        self.assertTrue(use_update.get("value"))

    def test_simple_lora_scale_clamps_and_keeps_status_in_sync(self) -> None:
        """Scale changes should update advanced scale and preserve next-run status."""

        with patch.object(
            simple_lora_wiring.gen_h,
            "update_lora_next_run_status",
            return_value=("No LoRA will be used.", {"value": False}),
        ):
            scale_update, status, use_update = simple_lora_wiring.sync_simple_lora_scale(
                3.5,
                "",
                "",
            )

        self.assertEqual(3.0, scale_update.get("value"))
        self.assertEqual("No LoRA will be used.", status)
        self.assertFalse(use_update.get("value"))

    def test_refresh_simple_lora_dropdown_keeps_valid_current_value(self) -> None:
        """Advanced refresh should refresh simple choices without a quick-tab button."""

        choices = [("None", ""), ("voice", "adapter")]
        with patch.object(simple_lora_wiring, "lora_dropdown_choices", return_value=choices):
            update = simple_lora_wiring.refresh_simple_lora_dropdown("adapter")

        self.assertEqual(choices, update.get("choices"))
        self.assertEqual("adapter", update.get("value"))


if __name__ == "__main__":
    unittest.main()
