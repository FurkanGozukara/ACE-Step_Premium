"""Unit tests for CLI task/model compatibility helpers."""

import unittest

from cli import (
    _filter_models_for_task,
    _task_model_requirement_message,
    _task_supported_by_model_config,
)


class CliTaskModelSupportTests(unittest.TestCase):
    """Verify advanced ACE-Step tasks are gated to pure base DiT configs."""

    def test_advanced_tasks_require_pure_base_config(self):
        """Extract, Lego, and Complete should reject SFT and turbo configs."""

        for task_type in ("extract", "lego", "complete"):
            with self.subTest(task_type=task_type):
                self.assertTrue(
                    _task_supported_by_model_config(task_type, "acestep-v15-xl-base")
                )
                self.assertFalse(
                    _task_supported_by_model_config(task_type, "acestep-v15-xl-sft")
                )
                self.assertFalse(
                    _task_supported_by_model_config(task_type, "acestep-v15-xl-turbo")
                )
                self.assertFalse(
                    _task_supported_by_model_config(task_type, "database-model")
                )

    def test_standard_tasks_allow_non_base_configs(self):
        """Text, cover, and repaint tasks should remain available everywhere."""

        for task_type in ("text2music", "cover", "repaint"):
            with self.subTest(task_type=task_type):
                self.assertTrue(
                    _task_supported_by_model_config(task_type, "acestep-v15-xl-sft")
                )
                self.assertTrue(
                    _task_supported_by_model_config(task_type, "acestep-v15-xl-turbo")
                )

    def test_filter_models_keeps_only_base_for_advanced_tasks(self):
        """Auto-selection should prefer only compatible Base models."""

        models = [
            "acestep-v15-xl-turbo",
            "acestep-v15-xl-sft",
            "acestep-v15-xl-base",
        ]

        self.assertEqual(["acestep-v15-xl-base"], _filter_models_for_task(models, "lego"))

    def test_requirement_message_mentions_base_only(self):
        """CLI validation text should match the actual compatibility rule."""

        message = _task_model_requirement_message("complete")

        self.assertIn("requires a base model config", message)
        self.assertNotIn("SFT", message)


if __name__ == "__main__":
    unittest.main()
