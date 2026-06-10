"""Contract tests for generation batch process controls."""

import ast
from pathlib import Path
import unittest


_BATCH_CONTROLS_PATH = Path(__file__).resolve().parent / "generation_tab_batch_extract_controls.py"


class GenerationTabBatchProcessControlsTests(unittest.TestCase):
    """Verify batch process controls start in a compact state."""

    def test_batch_process_accordion_defaults_closed(self):
        """Batch Process should be closed until the user opens it."""

        module = ast.parse(_BATCH_CONTROLS_PATH.read_text(encoding="utf-8"))
        func = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_batch_extract_controls"
        )
        accordion_call = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Accordion"
        )
        open_kw = next((kw for kw in accordion_call.keywords if kw.arg == "open"), None)

        self.assertIsNotNone(open_kw)
        self.assertIsInstance(open_kw.value, ast.Constant)
        self.assertFalse(open_kw.value.value)


if __name__ == "__main__":
    unittest.main()
