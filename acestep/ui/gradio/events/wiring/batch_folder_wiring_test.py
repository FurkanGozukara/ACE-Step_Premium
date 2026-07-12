"""Contract tests for Batch Folder Processing wiring."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


_WIRING_PATH = Path(__file__).resolve().parent / "batch_folder_wiring.py"


class BatchFolderWiringTests(unittest.TestCase):
    """Verify batch runs prepare current Generate Song values first."""

    def test_process_button_prepares_simple_values_before_batch_runner(self) -> None:
        """Batch duration, seed, model, and related values must match the GUI."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        register_fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "register_batch_folder_handlers"
        )
        self.assertIn("simple_page", [arg.arg for arg in register_fn.args.kwonlyargs])

        chained_calls = [
            node
            for node in ast.walk(register_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "then"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "click"
        ]
        process_chain = next(
            node
            for node in chained_calls
            if _keyword_name(node.func.value, "fn") == "prepare_simple_generation"
        )

        self.assertEqual("batch_wrapper", _keyword_name(process_chain, "fn"))
        self.assertEqual(
            "build_simple_prepare_inputs",
            _keyword_call_name(process_chain.func.value, "inputs"),
        )
        self.assertEqual(
            "build_simple_prepare_outputs",
            _keyword_call_name(process_chain.func.value, "outputs"),
        )


def _keyword_name(call: ast.Call, name: str) -> str | None:
    """Return a simple name used by a call keyword."""

    value = next((item.value for item in call.keywords if item.arg == name), None)
    return value.id if isinstance(value, ast.Name) else None


def _keyword_call_name(call: ast.Call, name: str) -> str | None:
    """Return the simple function name called by a call keyword."""

    value = next((item.value for item in call.keywords if item.arg == name), None)
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    return value.func.id


if __name__ == "__main__":
    unittest.main()
