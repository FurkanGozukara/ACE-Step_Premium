"""Contract tests for generation service-config row builders."""

import ast
import os
import tempfile
from pathlib import Path
import unittest

import gradio as gr

from acestep.model_downloader import get_models_dir
from acestep.ui.gradio.interfaces.generation_service_config_rows import (
    build_checkpoint_controls,
)

_ROWS_PATH = Path(__file__).resolve().parent / "generation_service_config_rows.py"


class _FakeDitHandler:
    def __init__(self, choices):
        self._choices = list(choices)

    def get_available_checkpoints(self):
        return list(self._choices)


class GenerationServiceConfigRowsTests(unittest.TestCase):
    """Verify service row builder contracts needed by the UI wiring."""

    def test_language_dropdown_is_explicitly_interactive(self):
        """Language selector should remain interactive for runtime language selection."""

        module = ast.parse(_ROWS_PATH.read_text(encoding="utf-8"))
        func = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_language_selector"
        )

        dropdown_calls = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Dropdown"
        ]
        self.assertTrue(dropdown_calls, "Expected a gr.Dropdown call in build_language_selector")

        interactive_kw = next(
            (
                kw
                for kw in dropdown_calls[0].keywords
                if kw.arg == "interactive"
            ),
            None,
        )
        self.assertIsNotNone(interactive_kw, "language_dropdown should set interactive explicitly")
        self.assertIsInstance(interactive_kw.value, ast.Constant)
        self.assertTrue(interactive_kw.value.value)

    def test_checkpoint_dropdown_defaults_to_install_local_models_dir(self):
        """Checkpoint dropdown should preselect the install-local models folder."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            expected = str(get_models_dir(project_root=tmp_dir))
            try:
                with gr.Blocks():
                    controls = build_checkpoint_controls(
                        dit_handler=_FakeDitHandler([expected]),
                        service_pre_initialized=False,
                        params={},
                    )
                self.assertEqual(controls["checkpoint_dropdown"].value, expected)
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

    def test_checkpoint_dropdown_preserves_initialized_value(self):
        """Initialized service state should keep its explicit checkpoint value."""

        expected = r"K:\custom\models"
        with gr.Blocks():
            controls = build_checkpoint_controls(
                dit_handler=_FakeDitHandler([expected]),
                service_pre_initialized=True,
                params={"checkpoint": expected},
            )
        self.assertEqual(controls["checkpoint_dropdown"].value, expected)


if __name__ == "__main__":
    unittest.main()
