"""Tests for legacy OpenRouter task routing helpers."""

import unittest

from acestep.constants import TASK_INSTRUCTIONS
from openrouter.task_routing import resolve_task_instruction, task_skips_lm


class OpenRouterTaskRoutingTests(unittest.TestCase):
    """Verify OpenRouter uses task-accurate DiT instructions and LM skips."""

    def test_resolve_task_instruction_uses_placeholder_safe_defaults(self):
        """Tasks without track fields should not pass raw placeholders to DiT."""

        expected = {
            "text2music": TASK_INSTRUCTIONS["text2music"],
            "cover": TASK_INSTRUCTIONS["cover"],
            "cover-nofsq": TASK_INSTRUCTIONS["cover"],
            "repaint": TASK_INSTRUCTIONS["repaint"],
            "extract": TASK_INSTRUCTIONS["extract_default"],
            "lego": TASK_INSTRUCTIONS["lego_default"],
            "complete": TASK_INSTRUCTIONS["complete_default"],
        }

        for task_type, expected_instruction in expected.items():
            with self.subTest(task_type=task_type):
                instruction = resolve_task_instruction(task_type)
                self.assertEqual(expected_instruction, instruction)
                self.assertNotIn("{", instruction)
                self.assertNotIn("}", instruction)

    def test_resolve_task_instruction_formats_track_context(self):
        """Advanced tasks should use supplied track context when present."""

        self.assertEqual(
            TASK_INSTRUCTIONS["extract"].format(TRACK_NAME="VOCALS"),
            resolve_task_instruction("extract", track_name="vocals"),
        )
        self.assertEqual(
            TASK_INSTRUCTIONS["lego"].format(TRACK_NAME="GUITAR"),
            resolve_task_instruction("lego", track_name="guitar"),
        )
        self.assertEqual(
            TASK_INSTRUCTIONS["complete"].format(TRACK_CLASSES="DRUMS | BASS"),
            resolve_task_instruction("complete", track_classes=["drums", "bass"]),
        )

    def test_task_skips_lm_matches_source_audio_only_tasks(self):
        """OpenRouter should skip LM/CoT for source-audio tasks handled by DiT."""

        for task_type in ("cover", "cover-nofsq", "repaint", "extract"):
            with self.subTest(task_type=task_type):
                self.assertTrue(task_skips_lm(task_type))

        for task_type in ("text2music", "lego", "complete"):
            with self.subTest(task_type=task_type):
                self.assertFalse(task_skips_lm(task_type))


if __name__ == "__main__":
    unittest.main()
