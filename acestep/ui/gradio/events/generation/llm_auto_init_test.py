"""Tests for on-demand LM initialization in Gradio generation actions."""

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.generation.llm_auto_init import ensure_llm_ready
from acestep.ui.gradio.events.generation.llm_format_actions import (
    handle_format_lyrics,
)


class _FakeLLMHandler:
    def __init__(self):
        self.llm_initialized = False
        self.last_init_params = None
        self.initialize_calls = []

    def get_default_lm_model(self):
        return "acestep-5Hz-lm-1.7B"

    def get_available_5hz_lm_models(self):
        return ["acestep-5Hz-lm-1.7B"]

    def initialize(self, **kwargs):
        self.initialize_calls.append(kwargs)
        self.llm_initialized = True
        return "LM ready", True


class LlmAutoInitTests(unittest.TestCase):
    def test_ensure_llm_ready_initializes_with_install_local_models_dir(self):
        """LM auto-init should use the current install-local models folder."""

        handler = _FakeLLMHandler()
        with tempfile.TemporaryDirectory() as tmp_dir:
            expected_models_dir = Path(tmp_dir) / "models"
            with patch.dict(os.environ, {"ACESTEP_PROJECT_ROOT": tmp_dir}, clear=False):
                with patch(
                    "acestep.ui.gradio.events.generation.llm_auto_init.get_global_gpu_config",
                    return_value=types.SimpleNamespace(recommended_backend="pt"),
                ), patch(
                    "acestep.ui.gradio.events.generation.llm_auto_init.resolve_lm_backend",
                    return_value="pt",
                ), patch(
                    "acestep.ui.gradio.events.generation.llm_auto_init.ensure_lm_model",
                    return_value=(True, "downloaded"),
                ) as ensure_model:
                    ok, status = ensure_llm_ready(
                        handler,
                        lm_model_path="",
                        backend=None,
                        device="auto",
                        offload_to_cpu=True,
                    )

        self.assertTrue(ok)
        self.assertIn("initialized automatically", status)
        ensure_model.assert_called_once_with(
            model_name="acestep-5Hz-lm-1.7B",
            checkpoints_dir=expected_models_dir,
        )
        self.assertEqual(
            handler.initialize_calls,
            [
                {
                    "checkpoint_dir": str(expected_models_dir),
                    "lm_model_path": "acestep-5Hz-lm-1.7B",
                    "backend": "pt",
                    "device": "auto",
                    "offload_to_cpu": True,
                    "dtype": None,
                }
            ],
        )
        self.assertEqual(
            handler.last_init_params,
            {
                "lm_model_path": "acestep-5Hz-lm-1.7B",
                "backend": "pt",
                "device": "auto",
                "offload_to_cpu": True,
            },
        )

    def test_ensure_llm_ready_uses_gpu_recommended_model_when_unset(self):
        """On-demand LM actions should follow the GPU-tier LM recommendation."""

        handler = _FakeLLMHandler()
        handler.get_available_5hz_lm_models = lambda: [
            "acestep-5Hz-lm-1.7B",
            "acestep-5Hz-lm-4B",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"ACESTEP_PROJECT_ROOT": tmp_dir}, clear=False):
                with patch(
                    "acestep.ui.gradio.events.generation.llm_auto_init.get_global_gpu_config",
                    return_value=types.SimpleNamespace(
                        recommended_backend="pt",
                        recommended_lm_model="acestep-5Hz-lm-4B",
                    ),
                ), patch(
                    "acestep.ui.gradio.events.generation.llm_auto_init.resolve_lm_backend",
                    return_value="pt",
                ), patch(
                    "acestep.ui.gradio.events.generation.llm_auto_init.ensure_lm_model",
                    return_value=(True, "downloaded"),
                ):
                    ok, _status = ensure_llm_ready(
                        handler,
                        lm_model_path="",
                        backend=None,
                        device="auto",
                        offload_to_cpu=False,
                    )

        self.assertTrue(ok)
        self.assertEqual(
            handler.initialize_calls[0]["lm_model_path"],
            "acestep-5Hz-lm-4B",
        )

    def test_ensure_llm_ready_skips_reinit_when_existing_runtime_matches(self):
        """Matching initialized LM runtime should be reused."""

        handler = _FakeLLMHandler()
        handler.llm_initialized = True
        handler.last_init_params = {
            "lm_model_path": "acestep-5Hz-lm-1.7B",
            "backend": "pt",
            "device": "auto",
            "offload_to_cpu": False,
        }

        with patch(
            "acestep.ui.gradio.events.generation.llm_auto_init.get_global_gpu_config",
            return_value=types.SimpleNamespace(recommended_backend="pt"),
        ), patch(
            "acestep.ui.gradio.events.generation.llm_auto_init.resolve_lm_backend",
            return_value="pt",
        ), patch(
            "acestep.ui.gradio.events.generation.llm_auto_init.ensure_lm_model",
        ) as ensure_model:
            ok, status = ensure_llm_ready(
                handler,
                lm_model_path="acestep-5Hz-lm-1.7B",
                backend="pt",
                device="auto",
                offload_to_cpu=False,
            )

        self.assertTrue(ok)
        self.assertEqual(status, "")
        ensure_model.assert_not_called()
        self.assertEqual(handler.initialize_calls, [])

    def test_handle_format_lyrics_auto_initializes_before_formatting(self):
        """Enhance Lyrics should auto-init the LM instead of failing immediately."""

        result = types.SimpleNamespace(
            success=True,
            lyrics='"Formatted lyrics"',
            bpm=120,
            duration=24,
            keyscale="C Major",
            language="en",
            timesignature="4/4",
            status_message="Formatted successfully",
        )

        with patch(
            "acestep.ui.gradio.events.generation.llm_format_actions.ensure_llm_ready",
            return_value=(True, "5Hz LM initialized automatically."),
        ), patch(
            "acestep.ui.gradio.events.generation.llm_format_actions.format_sample",
            return_value=result,
        ), patch(
            "acestep.ui.gradio.events.generation.llm_format_actions.clamp_duration_to_gpu_limit",
            return_value=24,
        ), patch(
            "gradio.Info",
        ), patch(
            "gradio.Warning",
        ):
            updates = handle_format_lyrics(
                llm_handler=object(),
                caption="caption",
                lyrics="lyrics",
                bpm=120,
                audio_duration=20,
                key_scale="C Major",
                time_signature="4/4",
                lm_temperature=0.8,
                lm_top_k=0,
                lm_top_p=0.95,
                constrained_decoding_debug=False,
                lm_model_path="acestep-5Hz-lm-1.7B",
                backend="pt",
                device="auto",
                offload_to_cpu=False,
            )

        self.assertEqual(updates[0], "Formatted lyrics")
        self.assertEqual(updates[1], 120)
        self.assertEqual(updates[2], 24)
        self.assertEqual(updates[3], "C Major")
        self.assertEqual(updates[4], "en")
        self.assertEqual(updates[5], "4/4")
        self.assertTrue(updates[6])
        self.assertIn("initialized automatically", updates[7])
        self.assertIn("Formatted successfully", updates[7])


if __name__ == "__main__":
    unittest.main()
