"""Unit tests for isolated generation worker argument handling."""

import inspect
import unittest

from acestep.ui.gradio.events.results.generation_progress import generate_with_progress
from acestep.ui.gradio.events.results.subprocess_worker import _build_generation_kwargs
from acestep.ui.gradio.premium_features import SIMPLE_MODEL_CHOICES, model_quality_defaults


class SubprocessWorkerArgumentTests(unittest.TestCase):
    """Verify subprocess worker kwargs stay aligned with generation controls."""

    def test_worker_kwargs_cover_required_generation_signature(self) -> None:
        """Worker kwargs should satisfy all required generation parameters."""

        kwargs = _build_generation_kwargs({"generation": {"captions": "test prompt"}})
        required_names = {
            name
            for name, parameter in inspect.signature(generate_with_progress).parameters.items()
            if name not in {"dit_handler", "llm_handler", "progress"}
            and parameter.default is inspect.Parameter.empty
        }

        self.assertFalse(required_names - set(kwargs))
        self.assertNotIn("dit_handler", kwargs)
        self.assertNotIn("llm_handler", kwargs)
        self.assertNotIn("progress", kwargs)

    def test_worker_preserves_model_preset_controls(self) -> None:
        """Worker kwargs should preserve controls used by model presets and Remix."""

        kwargs = _build_generation_kwargs(
            {
                "generation": {
                    "no_fsq": True,
                    "dcw_enabled": False,
                    "dcw_mode": "double",
                    "dcw_scaler": 0.12,
                    "dcw_high_scaler": 0.34,
                    "dcw_wavelet": "db4",
                }
            }
        )

        self.assertTrue(kwargs["no_fsq"])
        self.assertFalse(kwargs["dcw_enabled"])
        self.assertEqual(kwargs["dcw_mode"], "double")
        self.assertEqual(kwargs["dcw_scaler"], 0.12)
        self.assertEqual(kwargs["dcw_high_scaler"], 0.34)
        self.assertEqual(kwargs["dcw_wavelet"], "db4")
        self.assertIn("retake_variance", kwargs)
        self.assertIn("flow_edit_morph", kwargs)

    def test_worker_accepts_all_simple_model_presets(self) -> None:
        """Worker kwargs should accept every Simple-tab model preset payload."""

        for _label, model_path in SIMPLE_MODEL_CHOICES:
            generation = {"captions": "test prompt", **model_quality_defaults(model_path)}
            with self.subTest(model_path=model_path):
                kwargs = _build_generation_kwargs({"generation": generation})
                self.assertEqual(kwargs["inference_steps"], generation["inference_steps"])
                self.assertEqual(kwargs["dcw_enabled"], generation["dcw_enabled"])
                self.assertTrue(generation["generate_lm_audio_codes"])


if __name__ == "__main__":
    unittest.main()
