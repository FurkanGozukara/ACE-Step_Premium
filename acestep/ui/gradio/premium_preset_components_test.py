"""Tests for premium custom-preset component resolution."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.premium_features import get_preset_component_keys
from acestep.ui.gradio.premium_preset_components import (
    build_preset_component_map,
    preset_components_for_keys,
)
from acestep.ui.gradio.premium_preset_schema import SIMPLE_CREATE_COMPONENT_ALIASES


class PremiumPresetComponentMapTests(unittest.TestCase):
    """Verify preset schema keys resolve to concrete page components."""

    def test_component_map_resolves_every_schema_key(self) -> None:
        """Every preset key should resolve, including Generate Song aliases."""

        keys = get_preset_component_keys()
        generation_section = {
            key: object()
            for key in keys
            if key not in SIMPLE_CREATE_COMPONENT_ALIASES
        }
        simple_page = {
            simple_key: object()
            for simple_key in SIMPLE_CREATE_COMPONENT_ALIASES.values()
        }
        expected_alias_component = simple_page["simple_caption"]

        component_map = build_preset_component_map(
            generation_section=generation_section,
            simple_page=simple_page,
            training_section={},
            dataset_section={},
            batch_folder_section={},
        )
        components = preset_components_for_keys(component_map, keys)

        self.assertEqual(len(components), len(keys))
        self.assertIs(
            component_map["simple_create_caption"],
            expected_alias_component,
        )

    def test_missing_schema_key_raises_clear_error(self) -> None:
        """Missing component keys should fail before Gradio event wiring."""

        with self.assertRaisesRegex(KeyError, "missing_key"):
            preset_components_for_keys({}, ("missing_key",))


if __name__ == "__main__":
    unittest.main()

