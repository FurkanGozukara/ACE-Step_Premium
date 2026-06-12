"""Contract tests for Audio Processing progress display wiring."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[5]


class AudioProcessingProgressContractTests(unittest.TestCase):
    """Verify Audio Processing uses status outputs instead of stale progress overlays."""

    def test_process_file_hides_builtin_gradio_progress(self) -> None:
        """Process File should not leave a stale Gradio processing bar over outputs."""

        call = _click_call_for_component("ap_process_btn")

        self.assertEqual("hidden", _constant_keyword_value(call, "show_progress"))


def _click_call_for_component(component_key: str) -> ast.Call:
    """Return the click call for a component-map key in Audio Processing wiring."""

    module_path = _ROOT / "acestep/ui/gradio/events/wiring/audio_processing_wiring.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "click":
            continue
        target = node.func.value
        if not isinstance(target, ast.Subscript):
            continue
        if isinstance(target.slice, ast.Constant) and target.slice.value == component_key:
            return node
    raise AssertionError(f"Could not find click call for {component_key}")


def _constant_keyword_value(call: ast.Call, keyword_name: str) -> object:
    """Return a constant keyword value from an AST call."""

    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    raise AssertionError(f"Could not find constant keyword {keyword_name}")


if __name__ == "__main__":
    unittest.main()
