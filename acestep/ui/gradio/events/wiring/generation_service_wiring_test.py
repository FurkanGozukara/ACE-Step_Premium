"""Contract tests for generation service wiring."""

import ast
from pathlib import Path
import unittest

from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
)
from acestep.ui.gradio.events.wiring.generation_service_wiring import (
    _apply_config_path_change_with_simple_sync,
)


_WIRING_PATH = Path(__file__).resolve().parent / "generation_service_wiring.py"


class GenerationServiceWiringTests(unittest.TestCase):
    """Verify key event hooks are present in generation service wiring."""

    def test_registers_language_dropdown_change_handler(self):
        """Service wiring should attach a change handler for language dropdown."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        register_fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_generation_service_handlers"
        )

        found_language_change = False
        for node in ast.walk(register_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "change":
                continue
            if not isinstance(node.func.value, ast.Subscript):
                continue
            target = node.func.value
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "generation_section"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "language_dropdown"
            ):
                found_language_change = True
                break

        self.assertTrue(found_language_change, "language_dropdown.change handler was not found")

    def test_language_runtime_helper_exists(self):
        """Runtime language helper should exist for dropdown change wiring."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        function_names = {
            node.name for node in module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_apply_runtime_language", function_names)

    def test_config_path_change_applies_sft_quality_defaults(self):
        """Advanced model dropdown should apply LM-assisted SFT controls."""

        result = _apply_config_path_change_with_simple_sync(
            "acestep-v15-xl-sft",
            "Custom",
        )

        self.assertEqual(result[0].get("value"), 50)
        self.assertEqual(result[3].get("value"), 3.0)
        self.assertEqual(result[3].get("minimum"), 1.0)
        self.assertEqual(result[3].get("maximum"), 5.0)
        self.assertEqual(result[8].get("value"), True)
        self.assertEqual(result[9].get("value"), True)
        self.assertEqual(result[10].get("value"), False)
        self.assertEqual(result[11].get("value"), True)
        self.assertEqual(result[12].get("value"), True)
        self.assertEqual(result[13].get("value"), True)
        self.assertEqual(result[14].get("value"), False)
        self.assertEqual(result[15].get("value"), "double")
        self.assertEqual(result[16].get("value"), 0.0)
        self.assertEqual(result[17].get("value"), 0.0)
        self.assertEqual(result[18].get("value"), DEFAULT_PREMIUM_DIT_MODEL)
        self.assertEqual(result[19].get("value"), "ode")
        self.assertEqual(result[20].get("value"), "euler")
        self.assertEqual(result[21].get("value"), 0.0)
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), "")
        self.assertEqual(result[24].get("value"), "haar")

    def test_config_path_change_applies_base_direct_defaults(self):
        """Advanced model dropdown should reset LM/Think controls for Base."""

        result = _apply_config_path_change_with_simple_sync(
            "acestep-v15-xl-base",
            "Custom",
        )

        self.assertEqual(result[0].get("value"), 64)
        self.assertEqual(result[2].get("value"), False)
        self.assertEqual(result[3].get("value"), 3.0)
        self.assertEqual(result[3].get("minimum"), 1.0)
        self.assertEqual(result[3].get("maximum"), 5.0)
        self.assertEqual(result[8].get("value"), False)
        self.assertEqual(result[9].get("value"), False)
        self.assertEqual(result[10].get("value"), False)
        self.assertEqual(result[11].get("value"), False)
        self.assertEqual(result[12].get("value"), False)
        self.assertEqual(result[13].get("value"), False)
        self.assertEqual(result[14].get("value"), False)
        self.assertEqual(result[15].get("value"), "double")
        self.assertEqual(result[16].get("value"), 0.0)
        self.assertEqual(result[17].get("value"), 0.0)
        self.assertEqual(result[18].get("value"), DEFAULT_BASE_DIT_MODEL)
        self.assertEqual(result[19].get("value"), "ode")
        self.assertEqual(result[20].get("value"), "euler")
        self.assertEqual(result[21].get("value"), 0.0)
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), "")
        self.assertEqual(result[24].get("value"), "haar")

    def test_config_path_change_keeps_turbo_think_defaults(self):
        """Advanced model dropdown should keep LM/Think controls for Turbo."""

        result = _apply_config_path_change_with_simple_sync(
            "acestep-v15-xl-turbo",
            "Custom",
        )

        self.assertEqual(result[0].get("value"), 8)
        self.assertEqual(result[0].get("maximum"), 20)
        self.assertEqual(result[8].get("value"), True)
        self.assertEqual(result[9].get("value"), True)
        self.assertEqual(result[10].get("value"), True)
        self.assertEqual(result[11].get("value"), True)
        self.assertEqual(result[12].get("value"), True)
        self.assertEqual(result[13].get("value"), True)
        self.assertEqual(result[14].get("value"), True)
        self.assertEqual(result[15].get("value"), "double")
        self.assertEqual(result[16].get("value"), 0.02)
        self.assertEqual(result[17].get("value"), 0.06)
        self.assertEqual(result[18].get("value"), DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(result[19].get("value"), "ode")
        self.assertEqual(result[20].get("value"), "euler")
        self.assertEqual(result[21].get("value"), 0.0)
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), "")
        self.assertEqual(result[24].get("value"), "haar")


if __name__ == "__main__":
    unittest.main()
