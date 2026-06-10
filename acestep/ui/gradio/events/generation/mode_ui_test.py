"""Unit tests for mode_ui state-clearing behavior on mode switch.

Verifies that compute_mode_ui_updates correctly clears stale
text2music_audio_code_string and src_audio values when switching
between modes, preventing the state-leakage noise bug.

Also verifies that think_checkbox is restored to True when switching
back to Custom/Simple modes after Remix/Repaint forced it off.

Also verifies that task_type (a gr.State) is correctly set on every
mode switch so that stale "repaint" task_type cannot leak into Custom
mode generation and trigger the "requires source audio" error.
"""

import unittest
from types import SimpleNamespace

try:
    from acestep.constants import GENERATION_MODES_BASE, GENERATION_MODES_TURBO, MODE_TO_TASK_TYPE
    from acestep.ui.gradio.events.generation.mode_ui import compute_mode_ui_updates
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependency guard
    compute_mode_ui_updates = None
    _IMPORT_ERROR = exc

# Output indices for the two new state-clearing outputs
_IDX_TASK_TYPE = 5       # Index of task_type (gr.State) in compute_mode_ui_updates return tuple
_IDX_CUSTOM_MODE_GROUP = 1
_IDX_GENERATE_BTN = 2
_IDX_OPTIONAL_PARAMS = 4
_IDX_SRC_AUDIO_ROW = 6
_IDX_REPAINTING_GROUP = 7
_IDX_AUDIO_CODES_GROUP = 8
_IDX_TRACK_NAME = 9
_IDX_COMPLETE_TRACK_CLASSES = 10
_IDX_GENERATE_BTN_ROW = 11
_IDX_LOAD_FILE_COL = 15
_IDX_LOAD_FILE = 16
_IDX_AUDIO_CODES = 47
_IDX_SRC_AUDIO = 48
_IDX_FLOW_EDIT_COLUMN = 49
_IDX_FLOW_EDIT_MORPH = 50
_IDX_RUNTIME_OPTIONS_ROW = 51
_IDX_COMPOSITION_GUIDE = 52
_IDX_NO_FSQ_COLUMN = 53
_IDX_CUSTOM_HELP_GROUP = 54
_IDX_STRENGTH_VARIATION_ROW = 55
_IDX_THINK_CHECKBOX = 14
_IDX_GENERATION_MODE = 12
_IDX_PREVIOUS_GENERATION_MODE = 37
_IDX_REMIX_STRENGTH = 17
_IDX_COVER_NOISE = 18
_EXPECTED_TUPLE_LENGTH = 56
_IDX_BPM = 21
_IDX_KEY = 22
_IDX_TIMESIG = 23
_IDX_VOCAL_LANG = 24
_IDX_DURATION = 25
_IDX_AUTO_SCORE = 26
_IDX_AUTOGEN = 27
_IDX_AUTO_LRC = 28
_IDX_ANALYZE_BTN = 29
_IDX_REPAINTING_HEADER = 30
_IDX_REPAINTING_START = 31
_IDX_REPAINTING_END = 32


@unittest.skipIf(compute_mode_ui_updates is None,
                 f"compute_mode_ui_updates import unavailable: {_IMPORT_ERROR}")
class ModeUiStateClearingTests(unittest.TestCase):
    """Tests that mode switches clear stale UI state to prevent noise."""

    def test_tuple_length(self):
        """compute_mode_ui_updates should return exactly 56 elements."""
        result = compute_mode_ui_updates("Custom")
        self.assertEqual(len(result), _EXPECTED_TUPLE_LENGTH)

    def test_composition_guide_is_mode_specific(self):
        """Each generation mode should expose practical Composition guidance."""
        expectations = {
            "Simple": ("Simple", "plain-language", "does not use Source Audio"),
            "Custom": ("Custom", "LM Codes Hints", "Source Audio is ignored"),
            "Remix": ("Remix", "Remix Strength", "instrumental-only"),
            "Repaint": ("Repaint", "Repainting Start and End", "replacement section"),
            "Extract": ("Extract", "Track Name", "Caption, Lyrics, Reference Audio"),
            "Lego": ("Lego", "Track Name", "guide vocal"),
            "Complete": ("Complete", "track classes", "partial arrangement"),
        }

        for mode, expected_parts in expectations.items():
            with self.subTest(mode=mode):
                result = compute_mode_ui_updates(
                    mode,
                    previous_mode="Custom",
                    config_path="ACEStep_1_5_XL_Base_BF16",
                )
                guide_update = result[_IDX_COMPOSITION_GUIDE]
                guide_text = guide_update.get("value")
                for expected in expected_parts:
                    self.assertIn(expected, guide_text)

    def test_metadata_load_button_stays_hidden(self):
        """Mode changes should not reveal the deprecated metadata load button."""
        for mode in ("Simple", "Custom", "Remix", "Repaint", "Extract", "Lego", "Complete"):
            with self.subTest(mode=mode):
                result = compute_mode_ui_updates(
                    mode,
                    previous_mode="Custom",
                    config_path="ACEStep_1_5_XL_Base_BF16",
                )
                self.assertFalse(result[_IDX_LOAD_FILE_COL].get("visible"))
                self.assertFalse(result[_IDX_LOAD_FILE].get("visible"))

    def test_custom_mode_preserves_audio_codes(self):
        """In Custom mode, audio_codes textbox should be visible but not cleared."""
        result = compute_mode_ui_updates("Custom")
        codes_update = result[_IDX_AUDIO_CODES]
        # Should only set visibility, not clear the value
        self.assertTrue(codes_update.get("visible"))
        self.assertNotIn("value", codes_update)

    def test_remix_mode_clears_audio_codes(self):
        """Switching to Remix should clear the audio_codes textbox value."""
        result = compute_mode_ui_updates("Remix", previous_mode="Custom")
        codes_update = result[_IDX_AUDIO_CODES]
        self.assertEqual(codes_update.get("value"), "")
        self.assertFalse(codes_update.get("visible"))

    def test_simple_mode_clears_audio_codes(self):
        """Switching to Simple should clear the audio_codes textbox value."""
        result = compute_mode_ui_updates("Simple", previous_mode="Custom")
        codes_update = result[_IDX_AUDIO_CODES]
        self.assertEqual(codes_update.get("value"), "")

    def test_repaint_mode_clears_audio_codes(self):
        """Switching to Repaint should clear the audio_codes textbox value."""
        result = compute_mode_ui_updates("Repaint", previous_mode="Custom")
        codes_update = result[_IDX_AUDIO_CODES]
        self.assertEqual(codes_update.get("value"), "")

    def test_custom_mode_clears_src_audio(self):
        """Switching to Custom should clear src_audio (no source audio needed)."""
        result = compute_mode_ui_updates("Custom", previous_mode="Remix")
        src_update = result[_IDX_SRC_AUDIO]
        self.assertIsNone(src_update.get("value"))

    def test_simple_mode_clears_src_audio(self):
        """Switching to Simple should clear src_audio."""
        result = compute_mode_ui_updates("Simple", previous_mode="Remix")
        src_update = result[_IDX_SRC_AUDIO]
        self.assertIsNone(src_update.get("value"))

    def test_remix_mode_preserves_src_audio(self):
        """In Remix mode, src_audio should not be cleared (it's needed)."""
        result = compute_mode_ui_updates("Remix")
        src_update = result[_IDX_SRC_AUDIO]
        # Should be a no-op update (no value key)
        self.assertNotIn("value", src_update)

    def test_remix_mode_uses_cover_task_and_source_audio_controls(self):
        """Remix should keep the standard cover task and source-audio controls."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        result = compute_mode_ui_updates("Remix", llm_handler=llm_handler, previous_mode="Custom")
        self.assertEqual(result[_IDX_TASK_TYPE], "cover")
        self.assertTrue(result[_IDX_SRC_AUDIO_ROW].get("visible"))
        self.assertEqual(result[_IDX_AUDIO_CODES].get("value"), "")
        self.assertFalse(result[_IDX_AUDIO_CODES].get("visible"))
        self.assertNotIn("value", result[_IDX_SRC_AUDIO])
        self.assertTrue(result[_IDX_COVER_NOISE].get("visible"))
        self.assertEqual(result[_IDX_REMIX_STRENGTH].get("value"), 0.0)
        self.assertEqual(result[_IDX_COVER_NOISE].get("value"), 0.2)
        self.assertTrue(result[_IDX_REPAINTING_GROUP].get("visible"))
        self.assertIn("Remix Source Segment", result[_IDX_REPAINTING_HEADER].get("value"))
        self.assertEqual(result[_IDX_REPAINTING_START].get("label"), "Remix Source Start")
        self.assertEqual(result[_IDX_REPAINTING_END].get("label"), "Remix Source End")
        self.assertIn("sent into Remix", result[_IDX_REPAINTING_START].get("info"))
        self.assertIn("does not preserve", result[_IDX_REPAINTING_END].get("info"))

        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertFalse(think_update.get("value"))
        self.assertFalse(think_update.get("interactive"))

    def test_no_fsq_column_is_remix_only(self):
        """The no_fsq column should appear beside Remix Strength only in Remix."""
        remix_result = compute_mode_ui_updates("Remix", previous_mode="Custom")
        custom_result = compute_mode_ui_updates("Custom", previous_mode="Remix")
        repaint_result = compute_mode_ui_updates("Repaint", previous_mode="Remix")

        self.assertTrue(remix_result[_IDX_NO_FSQ_COLUMN].get("visible"))
        self.assertFalse(custom_result[_IDX_NO_FSQ_COLUMN].get("visible"))
        self.assertFalse(repaint_result[_IDX_NO_FSQ_COLUMN].get("visible"))

    def test_custom_top_control_row_visibility(self):
        """Custom should show the combined strength, help, Retake, and Edit row."""
        custom_result = compute_mode_ui_updates("Custom", previous_mode="Remix")
        remix_result = compute_mode_ui_updates("Remix", previous_mode="Custom")
        simple_result = compute_mode_ui_updates("Simple", previous_mode="Custom")

        self.assertTrue(custom_result[_IDX_STRENGTH_VARIATION_ROW].get("visible"))
        self.assertTrue(custom_result[_IDX_CUSTOM_HELP_GROUP].get("visible"))
        self.assertTrue(custom_result[_IDX_FLOW_EDIT_COLUMN].get("visible"))
        self.assertFalse(remix_result[_IDX_CUSTOM_HELP_GROUP].get("visible"))
        self.assertFalse(
            compute_mode_ui_updates("Repaint", previous_mode="Custom")[
                _IDX_CUSTOM_HELP_GROUP
            ].get("visible")
        )
        self.assertFalse(simple_result[_IDX_STRENGTH_VARIATION_ROW].get("visible"))

    def test_generation_modes_do_not_expose_raw_remix_as_top_level_mode(self):
        """Raw remix should be selected by no_fsq, not by a separate mode."""
        self.assertNotIn("Remix (Raw)", GENERATION_MODES_TURBO)
        self.assertNotIn("Remix (Raw)", GENERATION_MODES_BASE)
        self.assertNotIn("Remix (Raw)", MODE_TO_TASK_TYPE)

    def test_repaint_mode_preserves_src_audio(self):
        """In Repaint mode, src_audio should not be cleared (it's needed)."""
        result = compute_mode_ui_updates("Repaint")
        src_update = result[_IDX_SRC_AUDIO]
        self.assertNotIn("value", src_update)

    def test_repaint_mode_hides_and_disables_edit(self):
        """Repaint has its own local edit path, so flow-edit should be unavailable."""
        result = compute_mode_ui_updates("Repaint")

        self.assertFalse(result[_IDX_FLOW_EDIT_COLUMN].get("visible"))
        self.assertFalse(result[_IDX_FLOW_EDIT_MORPH].get("visible"))
        self.assertFalse(result[_IDX_FLOW_EDIT_MORPH].get("value"))

    def test_remix_mode_shows_edit(self):
        """Remix keeps the flow-edit overlay available."""
        result = compute_mode_ui_updates("Remix")

        self.assertTrue(result[_IDX_FLOW_EDIT_COLUMN].get("visible"))
        self.assertTrue(result[_IDX_FLOW_EDIT_MORPH].get("visible"))
        self.assertTrue(result[_IDX_FLOW_EDIT_MORPH].get("interactive"))

    def test_round_trip_remix_to_custom_clears_both(self):
        """Switching Remix -> Custom should clear both audio_codes and src_audio.

        analyze_btn in Remix mode writes codes to text2music_audio_code_string.
        These must be cleared when entering Custom mode so stale codes are not
        passed to the DiT (the root cause of garbled audio after tab-switching).
        """
        result = compute_mode_ui_updates("Custom", previous_mode="Remix")
        codes_update = result[_IDX_AUDIO_CODES]
        src_update = result[_IDX_SRC_AUDIO]
        # Codes from analyze_btn in Remix must be cleared
        self.assertEqual(codes_update.get("value"), "")
        self.assertTrue(codes_update.get("visible"))
        # src_audio should also be cleared
        self.assertIsNone(src_update.get("value"))

    def test_repaint_to_custom_clears_audio_codes(self):
        """Switching Repaint -> Custom should clear audio codes (analyze_btn contaminates them)."""
        result = compute_mode_ui_updates("Custom", previous_mode="Repaint")
        codes_update = result[_IDX_AUDIO_CODES]
        self.assertEqual(codes_update.get("value"), "")
        self.assertTrue(codes_update.get("visible"))

    def test_simple_to_custom_preserves_audio_codes(self):
        """Switching Simple -> Custom should NOT clear audio codes (Simple has no analyze_btn)."""
        result = compute_mode_ui_updates("Custom", previous_mode="Simple")
        codes_update = result[_IDX_AUDIO_CODES]
        self.assertNotIn("value", codes_update)
        self.assertTrue(codes_update.get("visible"))

    def test_round_trip_custom_to_remix_clears_codes(self):
        """Switching Custom -> Remix should clear stale audio codes."""
        result = compute_mode_ui_updates("Remix", previous_mode="Custom")
        codes_update = result[_IDX_AUDIO_CODES]
        self.assertEqual(codes_update.get("value"), "")

    def test_remix_mode_forces_think_checkbox_off(self):
        """Remix mode should force think_checkbox to False and non-interactive."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        result = compute_mode_ui_updates("Remix", llm_handler=llm_handler, previous_mode="Custom")
        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertFalse(think_update.get("value"))
        self.assertFalse(think_update.get("interactive"))

    def test_repaint_mode_forces_think_checkbox_off(self):
        """Repaint mode should force think_checkbox to False and non-interactive."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        result = compute_mode_ui_updates("Repaint", llm_handler=llm_handler, previous_mode="Custom")
        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertFalse(think_update.get("value"))
        self.assertFalse(think_update.get("interactive"))

    def test_remix_to_custom_restores_think_checkbox(self):
        """Switching Remix -> Custom should restore think_checkbox to True when LM is initialized.

        This is the core regression test for the tab-switch noise bug:
        think_checkbox was stuck at False after returning from Remix mode,
        causing the LLM to be skipped and producing garbled audio.
        """
        llm_handler = SimpleNamespace(llm_initialized=True)
        result = compute_mode_ui_updates("Custom", llm_handler=llm_handler, previous_mode="Remix")
        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertTrue(think_update.get("value"),
                        "think_checkbox must be restored to True when switching back to Custom mode")
        self.assertTrue(think_update.get("interactive"))

    def test_repaint_to_custom_restores_think_checkbox(self):
        """Switching Repaint -> Custom should restore think_checkbox to True."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        result = compute_mode_ui_updates("Custom", llm_handler=llm_handler, previous_mode="Repaint")
        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertTrue(think_update.get("value"))

    def test_remix_to_simple_restores_think_checkbox(self):
        """Switching Remix -> Simple should restore think_checkbox to True when LM is initialized."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        result = compute_mode_ui_updates("Simple", llm_handler=llm_handler, previous_mode="Remix")
        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertTrue(think_update.get("value"))

    def test_no_lm_still_allows_think_checkbox(self):
        """Without LM initialized, Think should stay available for delayed auto-init."""
        llm_handler = SimpleNamespace(llm_initialized=False)
        result = compute_mode_ui_updates("Custom", llm_handler=llm_handler, previous_mode="Remix")
        think_update = result[_IDX_THINK_CHECKBOX]
        self.assertTrue(think_update.get("value"))
        self.assertTrue(think_update.get("interactive"))

    def test_non_extract_modes_do_not_force_auto_fields_interactive(self):
        """Mode switches should not re-enable auto-managed metadata fields."""
        result = compute_mode_ui_updates("Remix", previous_mode="Custom")
        for idx in (_IDX_BPM, _IDX_KEY, _IDX_TIMESIG, _IDX_VOCAL_LANG, _IDX_DURATION):
            self.assertNotIn("interactive", result[idx])

    def test_leaving_extract_sets_auto_fields_non_interactive(self):
        """Leaving Extract should reset optional fields in locked auto state."""
        result = compute_mode_ui_updates("Custom", previous_mode="Extract")
        for idx in (_IDX_BPM, _IDX_KEY, _IDX_TIMESIG, _IDX_VOCAL_LANG, _IDX_DURATION):
            self.assertFalse(result[idx].get("interactive"))

    def test_sft_model_allows_non_extract_advanced_modes(self):
        """SFT models should allow Lego and Complete, while Extract stays Base-only."""

        cases = {
            "Lego": "lego",
            "Complete": "complete",
        }
        for mode, task_type in cases.items():
            with self.subTest(mode=mode):
                result = compute_mode_ui_updates(
                    mode,
                    previous_mode="Custom",
                    config_path="ACEStep_1_5_XL_SFT_BF16",
                )
                self.assertEqual(result[_IDX_TASK_TYPE], task_type)
                self.assertEqual(result[_IDX_PREVIOUS_GENERATION_MODE], mode)

    def test_sft_model_reverts_unsupported_extract_mode(self):
        """Extract should be visible but unavailable for SFT models."""

        result = compute_mode_ui_updates(
            "Extract",
            previous_mode="Custom",
            config_path="ACEStep_1_5_XL_SFT_BF16",
        )

        self.assertEqual(result[_IDX_TASK_TYPE], "text2music")
        self.assertEqual(result[_IDX_GENERATION_MODE].get("value"), "Custom")
        self.assertIn("not available", result[_IDX_GENERATION_MODE].get("info"))
        self.assertEqual(result[_IDX_PREVIOUS_GENERATION_MODE], "Custom")

    def test_extract_mode_applies_visible_controls(self):
        """Extract mode should show stem controls and hide Custom-only panels."""

        result = compute_mode_ui_updates(
            "Extract",
            previous_mode="Custom",
            config_path="ACEStep_1_5_XL_Base_BF16",
        )

        self.assertFalse(result[_IDX_CUSTOM_MODE_GROUP].get("visible"))
        self.assertFalse(result[_IDX_OPTIONAL_PARAMS].get("visible"))
        self.assertTrue(result[_IDX_SRC_AUDIO_ROW].get("visible"))
        self.assertFalse(result[_IDX_AUDIO_CODES_GROUP].get("visible"))
        self.assertTrue(result[_IDX_TRACK_NAME].get("visible"))
        self.assertTrue(result[_IDX_GENERATE_BTN_ROW].get("visible"))
        self.assertFalse(result[_IDX_GENERATE_BTN].get("interactive"))
        self.assertFalse(result[_IDX_RUNTIME_OPTIONS_ROW].get("visible"))
        self.assertFalse(result[_IDX_THINK_CHECKBOX].get("visible"))
        self.assertFalse(result[_IDX_AUTO_SCORE].get("visible"))
        self.assertFalse(result[_IDX_AUTOGEN].get("visible"))
        self.assertFalse(result[_IDX_AUTO_LRC].get("visible"))
        self.assertNotIn("visible", result[_IDX_ANALYZE_BTN])
        self.assertIn("Extract", result[_IDX_GENERATE_BTN].get("value"))

    def test_extract_mode_enables_generate_when_track_is_selected(self):
        """Extract should become runnable after a track name is selected."""

        result = compute_mode_ui_updates(
            "Extract",
            previous_mode="Custom",
            config_path="ACEStep_1_5_XL_Base_BF16",
            track_name="vocals",
        )

        self.assertTrue(result[_IDX_GENERATE_BTN].get("interactive"))

    def test_turbo_model_reverts_unsupported_extract_mode(self):
        """Unsupported advanced modes should be visible but not usable."""
        result = compute_mode_ui_updates(
            "Extract",
            previous_mode="Custom",
            config_path="ACEStep_1_5_XL_Turbo_BF16",
        )
        self.assertEqual(result[_IDX_TASK_TYPE], "text2music")
        self.assertEqual(result[_IDX_GENERATION_MODE].get("value"), "Custom")
        self.assertIn("not available", result[_IDX_GENERATION_MODE].get("info"))
        self.assertEqual(result[_IDX_PREVIOUS_GENERATION_MODE], "Custom")

    def test_complete_mode_shows_source_tracks_and_section_controls(self):
        """Complete should expose source audio, target tracks, and start/end controls."""

        result = compute_mode_ui_updates(
            "Complete",
            previous_mode="Custom",
            config_path="ACEStep_1_5_XL_Base_BF16",
        )

        self.assertTrue(result[_IDX_SRC_AUDIO_ROW].get("visible"))
        self.assertTrue(result[_IDX_COMPLETE_TRACK_CLASSES].get("visible"))
        self.assertTrue(result[_IDX_REPAINTING_GROUP].get("visible"))
        self.assertIn("Complete Section", result[_IDX_REPAINTING_HEADER].get("value"))
        self.assertEqual(result[_IDX_REPAINTING_START].get("label"), "Complete Start")
        self.assertEqual(result[_IDX_REPAINTING_END].get("label"), "Complete End")


@unittest.skipIf(compute_mode_ui_updates is None,
                 f"compute_mode_ui_updates import unavailable: {_IMPORT_ERROR}")
class ModeUiTaskTypeTests(unittest.TestCase):
    """Regression tests for task_type (gr.State) being correctly set on mode switches.

    These tests guard against the bug where switching from Repaint back to Custom
    mode left a stale ``task_type="repaint"`` value that caused the backend to raise
    "Task 'repaint' requires source audio, but none was provided."

    Because ``task_type`` is now a ``gr.State`` (not a hidden ``gr.Textbox``), the
    mode switch handler must return the raw string value directly rather than a
    ``gr.update()`` dict.  These tests confirm the returned value is a plain ``str``.
    """

    def test_repaint_to_custom_resets_task_type_to_text2music(self):
        """Switching Repaint → Custom must reset task_type to 'text2music'.

        This is the primary regression test for the bug described in the issue:
        going from Repaint back to Custom left task_type='repaint' in the UI
        state, causing generation to fail with "requires source audio".
        """
        result = compute_mode_ui_updates("Custom", previous_mode="Repaint")
        task_type_value = result[_IDX_TASK_TYPE]
        self.assertEqual(task_type_value, "text2music",
                         "task_type must be reset to 'text2music' when switching to Custom mode")
        self.assertIsInstance(task_type_value, str,
                              "task_type output must be a raw string (gr.State), not a gr.update() dict")

    def test_custom_mode_task_type_is_text2music(self):
        """Custom mode sets task_type to 'text2music'."""
        result = compute_mode_ui_updates("Custom")
        self.assertEqual(result[_IDX_TASK_TYPE], "text2music")

    def test_repaint_mode_task_type_is_repaint(self):
        """Repaint mode sets task_type to 'repaint'."""
        result = compute_mode_ui_updates("Repaint", previous_mode="Custom")
        self.assertEqual(result[_IDX_TASK_TYPE], "repaint")

    def test_remix_mode_task_type_is_cover(self):
        """Remix mode sets task_type to 'cover'."""
        result = compute_mode_ui_updates("Remix", previous_mode="Custom")
        self.assertEqual(result[_IDX_TASK_TYPE], "cover")

    def test_simple_mode_task_type_is_text2music(self):
        """Simple mode sets task_type to 'text2music'."""
        result = compute_mode_ui_updates("Simple", previous_mode="Custom")
        self.assertEqual(result[_IDX_TASK_TYPE], "text2music")

    def test_task_type_output_is_plain_string_not_dict(self):
        """task_type output must be a raw string, not a gr.update() dict.

        Using gr.State requires returning the raw value, not a gr.update() wrapper.
        Returning a gr.update() dict for a gr.State could cause Gradio to fail to
        update the state value, leaving stale values from previous modes.
        """
        for mode in ("Custom", "Repaint", "Remix", "Simple"):
            with self.subTest(mode=mode):
                result = compute_mode_ui_updates(mode)
                task_type_value = result[_IDX_TASK_TYPE]
                self.assertIsInstance(
                    task_type_value, str,
                    f"task_type for mode '{mode}' must be a plain string, got {type(task_type_value)}"
                )
                self.assertNotIsInstance(
                    task_type_value, dict,
                    f"task_type for mode '{mode}' must not be a gr.update() dict"
                )


try:
    from acestep.ui.gradio.events.generation.mode_ui import handle_extract_src_audio_change
    _EXTRACT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependency guard
    handle_extract_src_audio_change = None
    _EXTRACT_IMPORT_ERROR = exc


@unittest.skipIf(handle_extract_src_audio_change is None,
                 f"handle_extract_src_audio_change import unavailable: {_EXTRACT_IMPORT_ERROR}")
class ExtractSrcAudioDurationTests(unittest.TestCase):
    """Regression tests for issue #1118 — Gradio temp-path 'safe root' rejection.

    Gradio uploads land under the system temp dir (e.g. AppData\\Local\\Temp\\gradio
    on Windows), which is outside the project safe-root. The handler must read
    duration without invoking the training-module path-safety guard.
    """

    def test_returns_noop_for_non_source_locked_mode(self):
        """Modes without source-duration locking should return an empty update."""
        result = handle_extract_src_audio_change("/anywhere/file.wav", "Custom")
        self.assertNotIn("value", result)

    def test_returns_noop_for_empty_src_audio(self):
        """Empty src_audio should short-circuit without raising."""
        result = handle_extract_src_audio_change("", "Extract")
        self.assertNotIn("value", result)

    def test_reads_duration_from_gradio_temp_path_without_safe_path(self):
        """A Gradio temp path outside the project safe root must NOT raise."""
        from unittest.mock import patch, MagicMock
        fake_info = MagicMock(duration=42.7)
        gradio_temp_path = r"C:\Users\test\AppData\Local\Temp\gradio\abc\song.wav"
        with patch("soundfile.info", return_value=fake_info) as mock_info:
            result = handle_extract_src_audio_change(gradio_temp_path, "Extract")
            mock_info.assert_called_once_with(gradio_temp_path)
        # gr.update(value=...) returns a plain dict; assert directly.
        self.assertEqual(result.get("value"), 42.7)

    def test_reads_duration_from_video_media_helper(self):
        """Video source uploads should use decoded media duration for source-locked modes."""
        from unittest.mock import patch
        with patch(
            "acestep.ui.gradio.events.generation.mode_ui.media_audio_duration_seconds",
            return_value=19.5,
        ) as duration_mock:
            result = handle_extract_src_audio_change("clip.mp4", "Extract")
            duration_mock.assert_called_once_with("clip.mp4")
        self.assertEqual(result.get("value"), 19.5)

    def test_reads_duration_for_complete_mode(self):
        """Complete source uploads should auto-fill duration from the source."""
        from unittest.mock import patch
        with patch(
            "acestep.ui.gradio.events.generation.mode_ui.media_audio_duration_seconds",
            return_value=12.25,
        ) as duration_mock:
            result = handle_extract_src_audio_change("partial.wav", "Complete")
            duration_mock.assert_called_once_with("partial.wav")
        self.assertEqual(result.get("value"), 12.25)

    def test_reads_duration_from_latest_stale_upload_list(self):
        """Duration extraction should tolerate Gradio stale single-file lists."""
        from unittest.mock import patch
        with patch(
            "acestep.ui.gradio.events.generation.mode_ui.media_audio_duration_seconds",
            return_value=21.0,
        ) as duration_mock:
            result = handle_extract_src_audio_change(["old.wav", "clip.mp4"], "Extract")
            duration_mock.assert_called_once_with("clip.mp4")
        self.assertEqual(result.get("value"), 21.0)

    def test_swallows_invalid_audio_errors(self):
        """A bad/unreadable file should be logged-and-skipped, not raised."""
        from unittest.mock import patch
        with patch(
            "acestep.ui.gradio.events.generation.mode_ui.media_audio_duration_seconds",
            side_effect=RuntimeError("bad file"),
        ):
            result = handle_extract_src_audio_change("/tmp/bogus.wav", "Lego")
        self.assertNotIn("value", result)


if __name__ == "__main__":
    unittest.main()
