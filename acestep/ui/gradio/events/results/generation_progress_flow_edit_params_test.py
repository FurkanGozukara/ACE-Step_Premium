"""Tests for Gradio flow-edit parameter coercion."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acestep.ui.gradio.events.results import generation_progress
from acestep.ui.gradio.events.results.generation_progress_sequence_test import (
    _fake_result,
    _progress_args,
)


class GenerationProgressFlowEditParamsTest(unittest.TestCase):
    """Verify Gradio requests cannot pass invalid flow-edit params downstream."""

    def test_cot_language_detection_ignores_dropdown_language(self) -> None:
        """Manual CoT language detection should make the dropdown non-authoritative."""

        self.assertEqual(
            generation_progress._effective_vocal_language_for_generation("tr", False),
            "tr",
        )
        self.assertEqual(
            generation_progress._effective_vocal_language_for_generation("tr", True),
            "unknown",
        )

    def test_fractional_n_avg_becomes_one_before_generation(self) -> None:
        """The previous ``int(0.5)`` path should no longer pass ``0``."""

        captured = []

        def fake_sequence(_generate_music, _dit_handler, _llm_handler, *, params, **_kwargs):
            captured.append(params.flow_edit_n_avg)
            return _fake_result("100")

        with tempfile.TemporaryDirectory() as tmp:
            gpu_config = SimpleNamespace(
                save_memory_mode=False,
                max_duration_with_lm=600,
                max_duration_without_lm=600,
                gpu_memory_gb=24.0,
            )
            patches = [
                patch.object(generation_progress, "get_global_gpu_config", return_value=gpu_config),
                patch.object(generation_progress, "check_duration_limit", return_value=(True, "")),
                patch.object(generation_progress, "create_generation_run_dir", return_value=Path(tmp)),
                patch.object(generation_progress, "persist_generation_inputs", return_value={}),
                patch.object(
                    generation_progress,
                    "build_generation_manifest",
                    return_value=str(Path(tmp) / "manifest.json"),
                ),
                patch.object(generation_progress, "write_json", return_value=None),
                patch.object(
                    generation_progress,
                    "generate_sequential_songs",
                    side_effect=fake_sequence,
                ),
                patch("acestep.audio_utils.save_audio", side_effect=lambda output_path, **_: output_path),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                dit_handler = SimpleNamespace(last_init_params={"config_path": "xl-turbo"})
                llm_handler = SimpleNamespace(llm_initialized=False, last_init_params={})
                list(
                    generation_progress.generate_with_progress(
                        dit_handler,
                        llm_handler,
                        **_progress_args(flow_edit_n_avg=0.5),
                    )
                )

        self.assertEqual(captured, [1])


if __name__ == "__main__":
    unittest.main()
