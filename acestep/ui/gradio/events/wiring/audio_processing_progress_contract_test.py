"""Contract tests for Audio Processing progress display wiring."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[5]


class AudioProcessingProgressContractTests(unittest.TestCase):
    """Verify Audio Processing uses status outputs instead of stale progress overlays."""

    def test_process_file_prepare_step_hides_builtin_gradio_progress(self) -> None:
        """The immediate UI-prep click should not leave a stale progress overlay."""

        call = _click_call_for_component("ap_process_btn")

        self.assertEqual("hidden", _constant_keyword_value(call, "show_progress"))
        self.assertEqual([], _component_key_list_keyword_value(call, "show_progress_on"))

    def test_process_file_processing_step_targets_processed_outputs(self) -> None:
        """The long-running Process File job should show progress on result previews."""

        call = _assignment_call_for_name("process_event")

        self.assertEqual("then", call.func.attr)
        self.assertEqual("full", _constant_keyword_value(call, "show_progress"))
        self.assertEqual(
            ["ap_output_audio", "ap_output_video"],
            _component_key_list_keyword_value(call, "show_progress_on"),
        )
        self.assertEqual(
            "audio_processing_process",
            _constant_keyword_value(call, "api_name"),
        )


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


def _assignment_call_for_name(name: str) -> ast.Call:
    """Return the call assigned to a named variable in Audio Processing wiring."""

    module_path = _ROOT / "acestep/ui/gradio/events/wiring/audio_processing_wiring.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return node.value
    raise AssertionError(f"Could not find assignment call for {name}")


def _component_key_list_keyword_value(call: ast.Call, keyword_name: str) -> list[str]:
    """Return component-map keys from a list-valued keyword."""

    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        if not isinstance(keyword.value, ast.List):
            raise AssertionError(f"Keyword {keyword_name} is not a list")
        return [_component_key_from_subscript(item) for item in keyword.value.elts]
    raise AssertionError(f"Could not find list keyword {keyword_name}")


def _component_key_from_subscript(node: ast.AST) -> str:
    """Return the string key from ``audio_page['key']`` AST nodes."""

    if not isinstance(node, ast.Subscript):
        raise AssertionError(f"Expected component subscript, got {type(node).__name__}")
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    raise AssertionError("Expected a string component key")


if __name__ == "__main__":
    unittest.main()
