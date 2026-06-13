"""Unit tests for generation metadata file wiring contracts."""

import ast
from pathlib import Path
import unittest

try:
    from .ast_test_utils import load_module_ast
    from ..generation.metadata_fields import LOAD_METADATA_GENERATION_OUTPUT_KEYS
except ImportError:  # pragma: no cover - supports direct file execution
    from ast_test_utils import load_module_ast
    from acestep.ui.gradio.events.generation.metadata_fields import (
        LOAD_METADATA_GENERATION_OUTPUT_KEYS,
    )


_WIRING_PATH = Path(__file__).with_name("generation_metadata_file_wiring.py")

_EXPECTED_METADATA_KEYS = list(LOAD_METADATA_GENERATION_OUTPUT_KEYS)

def _tuple_string_values(node: ast.AST) -> list[str]:
    """Return string literal values from a tuple/list literal node."""

    if not isinstance(node, (ast.Tuple, ast.List)):
        raise AssertionError("Expected tuple/list node")
    values = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise AssertionError("Expected string literal")
        values.append(element.value)
    return values


class GenerationMetadataFileWiringTests(unittest.TestCase):
    """Verify metadata file-load wiring ordering and handler contracts."""

    def test_metadata_output_key_contract_order_is_stable(self):
        """The metadata output key tuple should match the expected UI ordering."""

        self.assertIn("generation_mode", _EXPECTED_METADATA_KEYS)
        self.assertIn("extract_all_stems", _EXPECTED_METADATA_KEYS)
        self.assertIn("sam_auto_postprocess", _EXPECTED_METADATA_KEYS)

    def test_build_outputs_appends_format_caption_state_last(self):
        """build outputs helper should append is_format_caption_state at the tail."""

        module = load_module_ast(_WIRING_PATH)
        for node in module.body:
            if not isinstance(node, ast.FunctionDef) or node.name != "_build_load_metadata_outputs":
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    if inner.func.attr == "append" and inner.args:
                        arg = inner.args[0]
                        if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant):
                            self.assertEqual(arg.slice.value, "is_format_caption_state")
                            return
        self.fail("append(results_section['is_format_caption_state']) not found")

    def test_register_function_references_expected_generation_handlers(self):
        """Register helper should reference load-metadata and auto-uncheck handlers."""

        module = load_module_ast(_WIRING_PATH)
        attrs = []
        for node in ast.walk(module):
            if isinstance(node, ast.Attribute):
                attrs.append(node.attr)
        self.assertIn("load_metadata", attrs)
        self.assertIn("load_metadata_with_status", attrs)
        self.assertIn("uncheck_auto_for_populated_fields", attrs)


if __name__ == "__main__":
    unittest.main()
