"""Contract tests for advanced DiT control construction."""

import ast
from pathlib import Path
import unittest


_DIT_CONTROLS_PATH = Path(__file__).resolve().parent / "generation_advanced_dit_controls.py"


class GenerationAdvancedDitControlsTests(unittest.TestCase):
    """Verify DiT controls honor model-specific UI config."""

    def test_inference_steps_uses_model_specific_maximum(self):
        """The initial inference slider should not expose Turbo as a 200-step model."""

        module = ast.parse(_DIT_CONTROLS_PATH.read_text(encoding="utf-8"))
        func = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_dit_controls"
        )
        assignment = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "inference_steps"
        )
        self.assertIsInstance(assignment.value, ast.Call)
        maximum_kw = next(
            kw for kw in assignment.value.keywords if kw.arg == "maximum"
        )
        self.assertIsInstance(maximum_kw.value, ast.Subscript)
        self.assertIsInstance(maximum_kw.value.slice, ast.Constant)
        self.assertEqual(maximum_kw.value.slice.value, "inference_steps_maximum")

    def test_dit_controls_no_longer_own_runtime_seed_controls(self):
        """Seed controls should live beside Auto LRC in the generation runtime row."""

        module = ast.parse(_DIT_CONTROLS_PATH.read_text(encoding="utf-8"))
        func = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_dit_controls"
        )
        return_node = next(node for node in ast.walk(func) if isinstance(node, ast.Return))
        self.assertIsInstance(return_node.value, ast.Dict)
        keys = {
            key.value
            for key in return_node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertNotIn("seed", keys)
        self.assertNotIn("random_seed_checkbox", keys)


if __name__ == "__main__":
    unittest.main()
