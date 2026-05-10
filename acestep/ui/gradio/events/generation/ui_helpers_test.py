"""Unit tests for small generation UI helper functions."""

import unittest
from typing import Any

from acestep.ui.gradio.events.generation.ui_helpers import (
    get_dcw_defaults_for_think,
    update_dcw_defaults_for_think,
    update_instruction_ui,
)


class _HandlerWithoutInstruction:
    """Handler stub that mirrors the lazy pre-init crash path."""

    def __getattr__(self, name: str) -> Any:
        """Raise when code tries to access runtime-only handler methods."""

        raise AttributeError(name)


class DcwDefaultTests(unittest.TestCase):
    """Validate Think-aware DCW default selection."""

    def test_think_mode_uses_think_dcw_defaults(self):
        """Think mode should use the LM-tuned DCW defaults."""
        defaults = get_dcw_defaults_for_think(True)
        self.assertEqual(defaults["mode"], "double")
        self.assertEqual(defaults["scaler"], 0.02)
        self.assertEqual(defaults["high_scaler"], 0.06)

    def test_non_think_mode_uses_original_dcw_defaults(self):
        """Non-Think mode should keep the existing pure-DiT DCW defaults."""
        defaults = get_dcw_defaults_for_think(False)
        self.assertEqual(defaults["mode"], "double")
        self.assertEqual(defaults["scaler"], 0.05)
        self.assertEqual(defaults["high_scaler"], 0.02)

    def test_update_dcw_defaults_returns_gradio_updates(self):
        """The event handler should return updates in component order."""
        mode_update, scaler_update, high_scaler_update = update_dcw_defaults_for_think(True)
        self.assertEqual(mode_update.get("value"), "double")
        self.assertEqual(scaler_update.get("value"), 0.02)
        self.assertEqual(high_scaler_update.get("value"), 0.06)


class InstructionUiTests(unittest.TestCase):
    """Validate instruction UI updates stay runtime-light."""

    def test_update_instruction_ui_does_not_require_runtime_handler_method(self):
        """Instruction text should render before the lazy DiT handler initializes."""

        result = update_instruction_ui(
            _HandlerWithoutInstruction(),
            "repaint",
            None,
            [],
        )

        self.assertEqual("Repaint the mask area based on the given conditions:", result)

    def test_update_instruction_ui_formats_complete_track_classes(self):
        """Complete mode should include selected track classes in the instruction."""

        result = update_instruction_ui(
            _HandlerWithoutInstruction(),
            "complete",
            None,
            ["drums", "bass"],
        )

        self.assertEqual("Complete the input track with DRUMS | BASS:", result)


if __name__ == "__main__":
    unittest.main()
