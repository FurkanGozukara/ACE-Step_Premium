"""Contract tests for non-queued media upload preview wiring."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


_ROOT = Path(__file__).resolve().parents[5]


class MediaUploadQueueContractTests(unittest.TestCase):
    """Verify upload-preview events bypass the global generation queue."""

    def test_audio_processing_upload_preview_bypasses_queue(self) -> None:
        """Audio Processing media upload preview should not wait behind queued jobs."""

        calls = _change_calls_for_component(
            "acestep/ui/gradio/events/wiring/audio_processing_wiring.py",
            "ap_single_file",
        )
        self.assertTrue(_all_calls_bypass_queue(calls))

    def test_sam_upload_previews_bypass_queue(self) -> None:
        """SAM media upload previews should not wait behind queued jobs."""

        calls = _change_calls_for_component(
            "acestep/ui/gradio/events/wiring/sam_audio_wiring.py",
            "sam_single_file",
        )
        self.assertTrue(_all_calls_bypass_queue(calls))
        calls = _change_calls_for_component(
            "acestep/ui/gradio/events/wiring/sam_audio_wiring.py",
            "sam_visual_mask_file",
        )
        self.assertTrue(_all_calls_bypass_queue(calls))

    def test_generation_media_upload_previews_bypass_queue(self) -> None:
        """Generation media upload preview/validation should not wait behind queued jobs."""

        for component_key in ("src_audio", "reference_audio", "lm_codes_audio_upload"):
            with self.subTest(component_key=component_key):
                calls = []
                calls.extend(
                    _change_calls_for_component(
                        "acestep/ui/gradio/events/wiring/generation_mode_wiring.py",
                        component_key,
                    )
                )
                calls.extend(
                    _change_calls_for_component(
                        "acestep/ui/gradio/events/wiring/generation_metadata_wiring.py",
                        component_key,
                    )
                )
                self.assertTrue(_all_calls_bypass_queue(calls), component_key)
                self.assertFalse(_has_self_output(calls, component_key), component_key)

    def test_generation_uploads_have_direct_preview_handlers(self) -> None:
        """Generation uploads should preview before slower follow-up work runs."""

        expected_previews = {
            (
                "acestep/ui/gradio/events/wiring/generation_mode_wiring.py",
                "src_audio",
            ): (
                "handle_src_audio_upload",
                [
                    "src_audio_preview",
                    "src_video_preview",
                    "audio_duration",
                    "src_audio_preview_original",
                ],
            ),
            (
                "acestep/ui/gradio/events/wiring/generation_metadata_wiring.py",
                "reference_audio",
            ): (
                "preview_audio_purpose_upload",
                ["reference_audio_preview", "reference_video_preview"],
            ),
            (
                "acestep/ui/gradio/events/wiring/generation_metadata_wiring.py",
                "lm_codes_audio_upload",
            ): (
                "preview_audio_purpose_upload",
                ["lm_codes_audio_preview", "lm_codes_video_preview"],
            ),
        }

        for (relative_path, component_key), (handler_name, outputs) in expected_previews.items():
            with self.subTest(component_key=component_key):
                calls = _change_calls_for_component(relative_path, component_key)
                self.assertTrue(
                    any(
                        _keyword_function_name(call, "fn") == handler_name
                        and _component_keys_for_keyword_or_variable(
                            relative_path,
                            call,
                            "outputs",
                        )
                        == outputs
                        for call in calls
                    )
                )

    def test_source_upload_finalize_progress_is_visible(self) -> None:
        """Source upload follow-up work should show visible progress."""

        calls = _calls_for_handler(
            "acestep/ui/gradio/events/wiring/generation_mode_wiring.py",
            "then",
            "finalize_src_audio_upload",
        )
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(_keyword_is_false(call, "queue"))
            self.assertEqual(_constant_keyword_value(call, "show_progress"), "full")

    def test_media_upload_files_have_isolated_gradio_state(self) -> None:
        """Media upload widgets should not preserve stale Gradio file values."""

        expected_uploads = {
            (
                "acestep/ui/gradio/interfaces/generation_tab_source_controls.py",
                "src_audio",
            ): (
                "advanced_source_audio_upload",
                "acestep-advanced-source-audio-upload",
            ),
            (
                "acestep/ui/gradio/interfaces/generation_tab_source_controls.py",
                "lm_codes_audio_upload",
            ): (
                "advanced_lm_codes_audio_upload",
                "acestep-advanced-lm-codes-audio-upload",
            ),
            (
                "acestep/ui/gradio/interfaces/generation_tab_secondary_controls.py",
                "reference_audio",
            ): (
                "advanced_reference_audio_upload",
                "acestep-advanced-reference-audio-upload",
            ),
            (
                "acestep/ui/gradio/pages/audio_processing_single_file_controls.py",
                "ap_single_file",
            ): (
                "audio_processing_single_upload",
                "acestep-audio-processing-single-upload",
            ),
            (
                "acestep/ui/gradio/pages/sam_audio_page_io.py",
                "sam_single_file",
            ): (
                "sam_single_upload",
                "acestep-sam-single-upload",
            ),
            (
                "acestep/ui/gradio/pages/sam_audio_page_io.py",
                "sam_visual_mask_file",
            ): (
                "sam_visual_mask_upload",
                "acestep-sam-visual-mask-upload",
            ),
        }

        for (relative_path, component_key), (gradio_key, elem_id) in expected_uploads.items():
            with self.subTest(component_key=component_key):
                call = _file_call_for_component(relative_path, component_key)
                self.assertEqual(_constant_keyword_value(call, "file_count"), "multiple")
                self.assertEqual(_constant_keyword_value(call, "key"), gradio_key)
                self.assertEqual(_constant_keyword_value(call, "elem_id"), elem_id)
                self.assertTrue(_has_empty_list_keyword(call, "preserved_by_key"))

    def test_upload_tabs_render_children_upfront(self) -> None:
        """Upload-heavy tabs should not lazy-render after another tab upload."""

        for tab_label in ("Audio Processing", "SAM Audio Segment"):
            with self.subTest(tab_label=tab_label):
                call = _tab_call_for_label("acestep/ui/gradio/premium_app.py", tab_label)
                self.assertTrue(_constant_keyword_value(call, "render_children"))


def _change_calls_for_component(relative_path: str, component_key: str) -> list[ast.Call]:
    """Return ``.change(...)`` calls for a specific component-map key."""

    module_path = _ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "change":
            continue
        target = node.func.value
        if not isinstance(target, ast.Subscript):
            continue
        if not isinstance(target.slice, ast.Constant):
            continue
        if target.slice.value == component_key:
            calls.append(node)
    return calls


def _calls_for_handler(
    relative_path: str,
    method_name: str,
    handler_name: str,
) -> list[ast.Call]:
    """Return method calls wired to a named handler function."""

    module_path = _ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != method_name:
            continue
        if _keyword_function_name(node, "fn") == handler_name:
            calls.append(node)
    return calls


def _file_call_for_component(relative_path: str, component_key: str) -> ast.Call:
    """Return the ``gr.File(...)`` call assigned to a component variable or map key."""

    module_path = _ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call) or not _is_gradio_call(node.value, "File"):
            continue
        for target in node.targets:
            if _target_matches_component_key(target, component_key):
                return node.value
    raise AssertionError(f"Could not find gr.File assignment for {component_key}")


def _tab_call_for_label(relative_path: str, tab_label: str) -> ast.Call:
    """Return the ``gr.Tab(...)`` call with a matching string label."""

    module_path = _ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_gradio_call(node, "Tab"):
            continue
        if not node.args:
            continue
        label = node.args[0]
        if isinstance(label, ast.Constant) and label.value == tab_label:
            return node
    raise AssertionError(f"Could not find gr.Tab for {tab_label}")


def _is_gradio_call(call: ast.Call, component_name: str) -> bool:
    """Return whether an AST call is ``gr.<component_name>(...)``."""

    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == component_name
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "gr"
    )


def _target_matches_component_key(target: ast.AST, component_key: str) -> bool:
    """Return whether an assignment target binds a component key."""

    if isinstance(target, ast.Name):
        return target.id == component_key
    if not isinstance(target, ast.Subscript):
        return False
    if not isinstance(target.slice, ast.Constant):
        return False
    return target.slice.value == component_key


def _constant_keyword_value(call: ast.Call, keyword_name: str) -> object:
    """Return a constant keyword value from an AST call."""

    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    raise AssertionError(f"Could not find constant keyword {keyword_name}")


def _has_empty_list_keyword(call: ast.Call, keyword_name: str) -> bool:
    """Return whether an AST call has an empty list keyword value."""

    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        return isinstance(keyword.value, ast.List) and not keyword.value.elts
    return False


def _all_calls_bypass_queue(calls: list[ast.Call]) -> bool:
    """Return whether every discovered upload ``.change`` call sets ``queue=False``."""

    if not calls:
        return False
    for call in calls:
        has_queue_false = False
        for keyword in call.keywords:
            if keyword.arg != "queue":
                continue
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                has_queue_false = True
        if not has_queue_false:
            return False
    return True


def _keyword_is_false(call: ast.Call, keyword_name: str) -> bool:
    """Return whether an AST call keyword is explicitly ``False``."""

    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        return isinstance(keyword.value, ast.Constant) and keyword.value.value is False
    return False


def _has_self_output(calls: list[ast.Call], component_key: str) -> bool:
    """Return whether any upload call writes back to the same file component."""

    for call in calls:
        for keyword in call.keywords:
            if keyword.arg != "outputs":
                continue
            if _node_contains_component_key(keyword.value, component_key):
                return True
    return False


def _keyword_function_name(call: ast.Call, keyword_name: str) -> str | None:
    """Return a simple function name assigned to a call keyword."""

    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        if isinstance(keyword.value, ast.Name):
            return keyword.value.id
        if isinstance(keyword.value, ast.Attribute):
            return keyword.value.attr
    return None


def _component_keys_for_keyword(call: ast.Call, keyword_name: str) -> list[str]:
    """Return component-map keys referenced by a list-valued call keyword."""

    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        if not isinstance(keyword.value, ast.List):
            return []
        return [
            item.slice.value
            for item in keyword.value.elts
            if isinstance(item, ast.Subscript)
            and isinstance(item.slice, ast.Constant)
            and isinstance(item.slice.value, str)
        ]
    return []


def _component_keys_for_keyword_or_variable(
    relative_path: str,
    call: ast.Call,
    keyword_name: str,
) -> list[str]:
    """Return component-map keys from an inline list or local list variable."""

    inline_keys = _component_keys_for_keyword(call, keyword_name)
    if inline_keys:
        return inline_keys

    keyword_value = _keyword_value(call, keyword_name)
    if not isinstance(keyword_value, ast.Name):
        return []

    module_path = _ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == keyword_value.id
            for target in node.targets
        ):
            continue
        return _component_keys_for_node(node.value)
    return []


def _component_keys_for_node(node: ast.AST) -> list[str]:
    """Return component-map keys referenced by a list AST node."""

    if not isinstance(node, ast.List):
        return []
    return [
        item.slice.value
        for item in node.elts
        if isinstance(item, ast.Subscript)
        and isinstance(item.slice, ast.Constant)
        and isinstance(item.slice.value, str)
    ]


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.AST | None:
    """Return a keyword AST value from a call."""

    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def _node_contains_component_key(node: ast.AST, component_key: str) -> bool:
    """Return whether an AST node references a component-map key."""

    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.slice, ast.Constant):
            continue
        if child.slice.value == component_key:
            return True
    return False


if __name__ == "__main__":
    unittest.main()
