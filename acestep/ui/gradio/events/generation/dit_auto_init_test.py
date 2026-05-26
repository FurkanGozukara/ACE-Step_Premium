"""Tests for on-demand DiT initialization in Gradio generation actions."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acestep.ui.gradio.events.generation.dit_auto_init import ensure_dit_ready
from acestep.ui.gradio.events.generation.llm_analysis_actions import analyze_src_audio


class _FakeDitHandler:
    """Small DiT handler test double for auto-initialization paths."""

    def __init__(self, *, model=None, init_ok: bool = True):
        """Store initial runtime state and initialization outcome."""

        self.model = model
        self.init_ok = init_ok
        self.initialize_calls = []
        self.convert_calls = []
        self.last_init_params = None

    def initialize_service(self, **kwargs):
        """Record initialization parameters and optionally mark the model loaded."""

        self.initialize_calls.append(kwargs)
        if self.init_ok:
            self.model = object()
            self.last_init_params = dict(kwargs)
        return "DiT ready", self.init_ok

    def convert_src_audio_to_codes(self, src_audio):
        """Record conversion calls and return serialized audio-code tokens."""

        self.convert_calls.append(src_audio)
        return "<|audio_code_1|><|audio_code_2|>"


class DitAutoInitTests(unittest.TestCase):
    """Verify foreground DiT auto-initialization behavior."""

    def test_ensure_dit_ready_initializes_missing_model(self):
        """Missing DiT runtime should initialize with selected UI options."""

        handler = _FakeDitHandler(model=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"ACESTEP_PROJECT_ROOT": tmp_dir}, clear=False):
                ok, status = ensure_dit_ready(
                    handler,
                    config_path="acestep-v15-xl-sft",
                    device="cuda",
                    use_flash_attention=True,
                    offload_to_cpu=True,
                    quantization="none",
                )

        self.assertTrue(ok)
        self.assertIn("DiT service initialized automatically", status)
        self.assertEqual(
            handler.initialize_calls,
            [
                {
                    "project_root": str(Path(tmp_dir).resolve()),
                    "config_path": "acestep-v15-xl-sft",
                    "device": "cuda",
                    "use_flash_attention": True,
                    "compile_model": False,
                    "offload_to_cpu": True,
                    "offload_dit_to_cpu": False,
                    "quantization": None,
                    "use_mlx_dit": True,
                }
            ],
        )

    def test_ensure_dit_ready_reuses_matching_loaded_model(self):
        """Matching loaded DiT runtime should not reinitialize."""

        handler = _FakeDitHandler(model=object())
        handler.last_init_params = {
            "config_path": "acestep-v15-xl-sft",
            "device": "cuda",
            "use_flash_attention": False,
            "offload_to_cpu": False,
            "offload_dit_to_cpu": False,
            "compile_model": False,
            "quantization": None,
            "use_mlx_dit": True,
        }

        ok, status = ensure_dit_ready(
            handler,
            config_path="acestep-v15-xl-sft",
            device="cuda",
            quantization="none",
        )

        self.assertTrue(ok)
        self.assertEqual(status, "")
        self.assertEqual(handler.initialize_calls, [])

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.clamp_duration_to_gpu_limit")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.ensure_llm_ready")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_auto_initializes_dit_before_conversion(
        self,
        understand_music_mock,
        ensure_llm_ready_mock,
        clamp_duration_mock,
    ):
        """Analyze should initialize DiT before converting source audio to codes."""

        handler = _FakeDitHandler(model=None)
        llm_handler = SimpleNamespace(llm_initialized=True)
        ensure_llm_ready_mock.return_value = (True, "")
        clamp_duration_mock.return_value = 30.0
        understand_music_mock.return_value = SimpleNamespace(
            success=True,
            status_message="analysis done",
            caption="caption",
            lyrics="lyrics",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4/4",
        )

        result = analyze_src_audio(
            handler,
            llm_handler,
            "source.wav",
            config_path="acestep-v15-xl-sft",
            device="auto",
            quantization="none",
        )

        self.assertEqual(handler.convert_calls, ["source.wav"])
        self.assertIn("DiT service initialized automatically", result[1])
        self.assertIn("analysis done", result[1])
        understand_music_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
