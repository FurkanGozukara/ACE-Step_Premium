"""Tests for Load Metadata generation-value restoration."""

import unittest

from acestep.ui.gradio.events.generation.metadata_fields import (
    LOAD_METADATA_GENERATION_OUTPUT_KEYS,
)
from acestep.ui.gradio.events.generation.metadata_restore_values import (
    metadata_values_from_payload,
)
from acestep.ui.gradio.events.generation.metadata_restore_document import (
    generation_payload_from_document,
)


def _restore_map(payload):
    """Return restored metadata values keyed by generation component name."""

    return dict(
        zip(LOAD_METADATA_GENERATION_OUTPUT_KEYS, metadata_values_from_payload(payload))
    )


class MetadataRestoreValuesTests(unittest.TestCase):
    """Verify restored metadata cannot leak Remix-only retention into text generation."""

    def test_text2music_clears_saved_remix_retention(self):
        values = _restore_map(
            {
                "generation_params": {
                    "task_type": "text2music",
                    "cover_noise_strength": 0.97,
                }
            }
        )

        self.assertEqual(values["task_type"], "text2music")
        self.assertEqual(values["cover_noise_strength"], 0.0)

    def test_cover_preserves_saved_remix_retention(self):
        values = _restore_map(
            {
                "generation_params": {
                    "task_type": "cover",
                    "cover_noise_strength": 0.42,
                }
            }
        )

        self.assertEqual(values["task_type"], "cover")
        self.assertEqual(values["cover_noise_strength"], 0.42)

    def test_compile_threads_restore_from_saved_runtime(self):
        payload = generation_payload_from_document(
            {
                "generation": {},
                "service": {"compile_model": True, "compile_threads": 17},
            }
        )

        values = _restore_map(payload)

        self.assertTrue(values["compile_model_checkbox"])
        self.assertEqual(values["compile_threads_slider"], 17)

    def test_compile_threads_restore_is_clamped(self):
        values = _restore_map(
            {"ui_runtime_settings": {"compile_threads_slider": 99}}
        )

        self.assertEqual(values["compile_threads_slider"], 32)


if __name__ == "__main__":
    unittest.main()
