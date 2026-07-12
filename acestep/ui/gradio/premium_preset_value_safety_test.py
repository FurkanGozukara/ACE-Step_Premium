"""Tests for component-aware custom preset value safety."""

from __future__ import annotations

import unittest

import gradio as gr

from acestep.ui.gradio.premium_preset_value_safety import (
    coerce_preset_value,
    component_specs_from_components,
)


class PremiumPresetValueSafetyTests(unittest.TestCase):
    """Verify list-valued Gradio controls survive preset round trips."""

    def test_checkbox_group_keeps_multiple_selected_values(self) -> None:
        component = gr.CheckboxGroup(
            choices=["woodwinds", "vocals", "drums"],
            value=[],
        )
        specs = component_specs_from_components(["tracks"], [component])

        self.assertTrue(specs["tracks"]["multiselect"])
        self.assertEqual(
            coerce_preset_value("tracks", ["vocals", "drums"], specs),
            ["vocals", "drums"],
        )

    def test_checkbox_group_keeps_explicit_empty_selection(self) -> None:
        component = gr.CheckboxGroup(
            choices=["woodwinds", "vocals", "drums"],
            value=[],
        )
        specs = component_specs_from_components(["tracks"], [component])

        self.assertEqual(coerce_preset_value("tracks", [], specs), [])


if __name__ == "__main__":
    unittest.main()
