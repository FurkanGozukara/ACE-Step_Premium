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
    _apply_tier_change_with_simple_quantization,
    _negative_prompt_update_for_model,
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

    def test_service_init_wiring_has_fifteen_handler_outputs(self):
        """Initialize Service outputs must match init_service_wrapper's tuple."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        register_fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "register_generation_service_handlers"
        )
        init_assignment = next(
            node
            for node in register_fn.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "init_event"
                for target in node.targets
            )
        )
        outputs_keyword = next(
            keyword
            for keyword in init_assignment.value.keywords
            if keyword.arg == "outputs"
        )

        self.assertIsInstance(outputs_keyword.value, ast.List)
        self.assertEqual(15, len(outputs_keyword.value.elts))

    def test_language_runtime_helper_exists(self):
        """Runtime language helper should exist for dropdown change wiring."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        function_names = {
            node.name for node in module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_apply_runtime_language", function_names)

    def test_advanced_outputs_folder_buttons_are_wired(self):
        """Both Advanced output-folder buttons should call the shared opener."""

        source = _WIRING_PATH.read_text(encoding="utf-8")
        self.assertIn('generation_section["open_outputs_folder_btn"].click', source)
        self.assertIn(
            'generation_section["generate_row_open_outputs_folder_btn"].click',
            source,
        )
        self.assertGreaterEqual(source.count("fn=open_outputs_folder"), 2)

    def test_config_path_model_sync_uses_user_input_event(self):
        """Model sync must not run for function-driven dropdown updates."""

        source = _WIRING_PATH.read_text(encoding="utf-8")
        self.assertIn('generation_section["config_path"].input', source)
        self.assertNotIn('generation_section["config_path"].change', source)

    def test_config_path_change_disables_turbo_negative_prompt(self):
        """Advanced model changes should update the negative prompt state."""

        turbo_update = _negative_prompt_update_for_model("acestep-v15-xl-turbo")
        sft_update = _negative_prompt_update_for_model("acestep-v15-xl-sft")

        self.assertFalse(turbo_update["interactive"])
        self.assertIn("CFG 1", turbo_update["info"])
        self.assertTrue(sft_update["interactive"])

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
        self.assertEqual(result[10].get("value"), True)
        self.assertEqual(result[11].get("value"), False)
        self.assertEqual(result[12].get("value"), True)
        self.assertEqual(result[13].get("value"), False)
        self.assertEqual(result[14].get("value"), False)
        self.assertEqual(result[15].get("value"), False)
        self.assertEqual(result[16].get("value"), "double")
        self.assertEqual(result[17].get("value"), 0.0)
        self.assertEqual(result[18].get("value"), 0.0)
        self.assertEqual(result[19].get("value"), DEFAULT_PREMIUM_DIT_MODEL)
        self.assertEqual(result[20].get("value"), "ode")
        self.assertEqual(result[21].get("value"), "heun")
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), 0.0)
        self.assertEqual(result[24].get("value"), "")
        self.assertEqual(result[25].get("value"), "haar")

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
        self.assertEqual(result[10].get("value"), True)
        self.assertEqual(result[11].get("value"), False)
        self.assertEqual(result[12].get("value"), False)
        self.assertEqual(result[13].get("value"), False)
        self.assertEqual(result[14].get("value"), False)
        self.assertEqual(result[15].get("value"), False)
        self.assertEqual(result[16].get("value"), "double")
        self.assertEqual(result[17].get("value"), 0.0)
        self.assertEqual(result[18].get("value"), 0.0)
        self.assertEqual(result[19].get("value"), DEFAULT_BASE_DIT_MODEL)
        self.assertEqual(result[20].get("value"), "ode")
        self.assertEqual(result[21].get("value"), "heun")
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), 0.0)
        self.assertEqual(result[24].get("value"), "")
        self.assertEqual(result[25].get("value"), "haar")

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
        self.assertEqual(result[13].get("value"), False)
        self.assertEqual(result[14].get("value"), False)
        self.assertEqual(result[15].get("value"), True)
        self.assertEqual(result[16].get("value"), "double")
        self.assertEqual(result[17].get("value"), 0.02)
        self.assertEqual(result[18].get("value"), 0.06)
        self.assertEqual(result[19].get("value"), DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(result[20].get("value"), "ode")
        self.assertEqual(result[21].get("value"), "heun")
        self.assertEqual(result[22].get("value"), 0.0)
        self.assertEqual(result[23].get("value"), 0.0)
        self.assertEqual(result[24].get("value"), "")
        self.assertEqual(result[25].get("value"), "haar")

    def test_advanced_tier_change_mirrors_simple_tier_and_quantization(self):
        """Advanced GPU presets should keep the Generate Song tab synchronized."""

        class LlmHandler:
            @staticmethod
            def get_available_5hz_lm_models():
                return []

        result = _apply_tier_change_with_simple_quantization(
            "unlimited",
            LlmHandler(),
        )

        self.assertEqual(len(result), 13)
        self.assertEqual(result[4].get("value"), result[3].get("value"))
        self.assertEqual(result[12].get("value"), "unlimited")


if __name__ == "__main__":
    unittest.main()
