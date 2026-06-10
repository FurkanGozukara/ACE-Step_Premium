"""Tests for extracted ``generate_music`` orchestration behavior.

The module loads ``acestep.core.generation.handler.generate_music`` directly
from file to avoid package import side effects and validates orchestration
ordering, readiness short-circuiting, and failure payload handling.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import torch


def _load_generate_music_module():
    """Load ``generate_music.py`` from disk for isolated mixin tests.

    Returns:
        types.ModuleType: Loaded module object for
        ``acestep.core.generation.handler.generate_music``.

    Raises:
        FileNotFoundError: If the target module file is missing.
        ImportError: If module loading fails.
    """
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    package_paths = {
        "acestep": repo_root / "acestep",
        "acestep.core": repo_root / "acestep" / "core",
        "acestep.core.generation": repo_root / "acestep" / "core" / "generation",
        "acestep.core.generation.handler": repo_root / "acestep" / "core" / "generation" / "handler",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        sys.modules[package_name] = package_module
    module_path = Path(__file__).with_name("generate_music.py")
    spec = importlib.util.spec_from_file_location(
        "acestep.core.generation.handler.generate_music",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GENERATE_MUSIC_MODULE = _load_generate_music_module()
GenerateMusicMixin = GENERATE_MUSIC_MODULE.GenerateMusicMixin


class _Host(GenerateMusicMixin):
    """Minimal host implementing ``generate_music`` helper dependencies.

    The host captures helper calls in ``self.calls`` and returns deterministic
    payloads so tests can assert orchestration sequencing and return behavior.
    """

    def __init__(
        self,
        offload_to_cpu: bool = False,
        is_turbo: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: str | None = None,
    ):
        """Initialize deterministic state and stub payloads for orchestration tests."""
        self.model = object()
        self.vae = object()
        self.text_tokenizer = object()
        self.text_encoder = object()
        self.offload_to_cpu = offload_to_cpu
        self.offload_dit_to_cpu = offload_dit_to_cpu
        self.quantization = quantization
        self._is_turbo = is_turbo
        self.last_init_params = {
            "config_path": "ACEStep_1_5_XL_Turbo_BF16" if is_turbo else "ACEStep_1_5_XL_Base_BF16",
        }
        self.sample_rate = 48000
        self.calls: Dict[str, Any] = {}
        self._final_payload = {"audios": [{"tensor": torch.zeros(1, 4), "sample_rate": 48000}], "success": True}
        self._readiness_error = {
            "audios": [],
            "status_message": "not ready",
            "extra_outputs": {},
            "success": False,
            "error": "Model not fully initialized",
        }

    def is_turbo_model(self):
        """Return whether this host simulates a turbo model."""
        return self._is_turbo

    def _resolve_generate_music_progress(self, progress):
        """Return provided callback or deterministic no-op callback."""
        self.calls["_resolve_generate_music_progress"] = bool(progress)
        if progress is not None:
            return progress

        def _noop(*_args, **_kwargs):
            """Ignore progress updates in tests."""
            return None

        return _noop

    def _validate_generate_music_readiness(self):
        """Return deterministic readiness error payload."""
        self.calls["_validate_generate_music_readiness"] = True
        return self._readiness_error

    def _resolve_generate_music_task(self, **kwargs):
        """Capture task resolution args and return deterministic task/instruction."""
        self.calls["_resolve_generate_music_task"] = kwargs
        return kwargs["task_type"], kwargs["instruction"]

    def _prepare_generate_music_runtime(self, **kwargs):
        """Capture runtime args and return deterministic runtime state."""
        self.calls["_prepare_generate_music_runtime"] = kwargs
        return {
            "actual_batch_size": 1,
            "actual_seed_list": [77],
            "seed_value_for_ui": 77,
            "actual_retake_seed_list": None,
            "retake_seed_value_for_ui": "",
            "audio_duration": kwargs["audio_duration"],
            "repainting_end": kwargs["repainting_end"],
        }

    def _prepare_reference_and_source_audio(self, **kwargs):
        """Capture audio-prepare args and return deterministic prepared state."""
        self.calls["_prepare_reference_and_source_audio"] = kwargs
        return [[torch.zeros(2, 10)]], None, None

    def _prepare_generate_music_service_inputs(self, **kwargs):
        """Capture service-input args and return deterministic payload."""
        self.calls["_prepare_generate_music_service_inputs"] = kwargs
        return {"should_return_intermediate": True}

    def _run_generate_music_service_with_progress(self, **kwargs):
        """Capture service execution args and return deterministic model outputs."""
        self.calls["_run_generate_music_service_with_progress"] = kwargs
        return {
            "outputs": {
                "target_latents": torch.ones(1, 4, 3),
                "time_costs": {"total_time_cost": 1.0, "diffusion_per_step_time_cost": 0.1},
            },
            "infer_steps_for_progress": 8,
        }

    def _prepare_generate_music_decode_state(self, **kwargs):
        """Capture decode-state args and return deterministic latents/costs."""
        self.calls["_prepare_generate_music_decode_state"] = kwargs
        return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

    def _decode_generate_music_pred_latents(self, **kwargs):
        """Capture decode args and return deterministic decode outputs."""
        self.calls["_decode_generate_music_pred_latents"] = kwargs
        return torch.ones(1, 2, 8), torch.ones(1, 4, 3), {"total_time_cost": 2.0}

    def _build_generate_music_success_payload(self, **kwargs):
        """Capture payload-builder args and return deterministic success payload."""
        self.calls["_build_generate_music_success_payload"] = kwargs
        return self._final_payload

    def _empty_cache(self):
        """No-op cache clear for test host."""


class GenerateMusicMixinTests(unittest.TestCase):
    """Verify top-level ``generate_music`` orchestration behavior."""

    def test_generate_music_returns_success_payload_from_builder(self):
        """It executes helper stages and returns the payload builder result."""
        host = _Host()
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            inference_steps=8,
            guidance_scale=6.5,
            use_random_seed=False,
            seed=77,
            task_type="text2music",
        )
        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.calls["_prepare_generate_music_runtime"]["seed"], 77)
        self.assertEqual(host.calls["_run_generate_music_service_with_progress"]["guidance_scale"], 6.5)
        self.assertEqual(host.calls["_prepare_generate_music_decode_state"]["infer_steps_for_progress"], 8)

    def test_generate_music_returns_readiness_error_when_components_missing(self):
        """It short-circuits with readiness payload when required models are missing."""
        host = _Host()
        host.model = None
        out = host.generate_music(captions="cap", lyrics="lyr")
        self.assertEqual(out, host._readiness_error)
        self.assertTrue(host.calls["_validate_generate_music_readiness"])
        self.assertNotIn("_prepare_generate_music_runtime", host.calls)

    def test_generate_music_returns_error_payload_on_exception(self):
        """It catches orchestration errors and returns standardized failure payload."""
        host = _Host()

        def _raise_error(**_kwargs):
            """Raise deterministic runtime failure for exception-path validation."""
            raise RuntimeError("boom")

        host._prepare_reference_and_source_audio = _raise_error
        out = host.generate_music(captions="cap", lyrics="lyr")
        self.assertFalse(out["success"])
        self.assertEqual(out["error"], "boom")
        self.assertIn("Error: boom", out["status_message"])

    def test_generate_music_normalizes_invalid_shift(self):
        """Invalid handler shift should not reach service generation."""
        host = _Host()
        out = host.generate_music(captions="cap", lyrics="lyr", shift=0.0)
        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.calls["_run_generate_music_service_with_progress"]["shift"], 1.0)

    def test_generate_music_rejects_non_finite_timesteps(self):
        """Invalid custom timesteps should return a clear error payload."""
        host = _Host()
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            timesteps=[1.0, float("nan"), 0.0],
        )
        self.assertFalse(out["success"])
        self.assertIn("Custom timesteps", out["error"])

    def test_repaint_forwards_cached_source_latents_to_service(self):
        """Generated-source repaint should only override the source-latent input."""
        host = _Host()
        source_latents = torch.ones(4, 3)

        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            task_type="repaint",
            repainting_start=1.0,
            repainting_end=2.0,
            source_repaint_latents=source_latents,
        )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(
            "repaint",
            host.calls["_run_generate_music_service_with_progress"]["task_type"],
        )
        self.assertIs(
            source_latents,
            host.calls["_run_generate_music_service_with_progress"]["source_repaint_latents"],
        )

    def test_complete_locks_duration_to_source_audio(self):
        """Complete should use the source audio length instead of user duration."""
        host = _Host()

        def _prepare_audio(**kwargs):
            """Return a three-second prepared source audio tensor."""
            host.calls["_prepare_reference_and_source_audio"] = kwargs
            return [[torch.zeros(2, 10)]], torch.ones(2, 48000 * 3), None

        host._prepare_reference_and_source_audio = _prepare_audio

        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            task_type="complete",
            src_audio="source.wav",
            audio_duration=99.0,
        )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(
            host.calls["_prepare_generate_music_service_inputs"]["audio_duration"],
            3.0,
        )

    @patch.object(GENERATE_MUSIC_MODULE, "apply_repaint_waveform_splice")
    def test_complete_with_range_splices_generated_section_back_into_source(self, splice_mock):
        """Complete ranges should merge generated audio into the original source waveform."""
        host = _Host()
        source_audio = torch.full((2, 8), 3.0)
        spliced = torch.full((1, 2, 8), 7.0)
        splice_mock.return_value = spliced

        def _prepare_audio(**kwargs):
            """Return a prepared source waveform for Complete splicing."""
            host.calls["_prepare_reference_and_source_audio"] = kwargs
            return [[torch.zeros(2, 10)]], source_audio, None

        def _prepare_service_inputs(**kwargs):
            """Return repaint-range service inputs for Complete."""
            host.calls["_prepare_generate_music_service_inputs"] = kwargs
            return {
                "should_return_intermediate": True,
                "repainting_start_batch": [1.0],
                "repainting_end_batch": [2.0],
                "target_wavs_tensor": torch.ones(1, 2, 8),
            }

        host._prepare_reference_and_source_audio = _prepare_audio
        host._prepare_generate_music_service_inputs = _prepare_service_inputs

        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            task_type="complete",
            src_audio="source.wav",
            repainting_start=1.0,
            repainting_end=2.0,
        )

        self.assertEqual(out, host._final_payload)
        splice_mock.assert_called_once()
        self.assertIs(source_audio, splice_mock.call_args.kwargs["src_wavs"])
        self.assertEqual(
            10,
            host.calls["_run_generate_music_service_with_progress"]["repaint_crossfade_frames"],
        )
        self.assertEqual(0.0, splice_mock.call_args.kwargs["crossfade_duration"])
        self.assertIs(
            spliced,
            host.calls["_build_generate_music_success_payload"]["pred_wavs"],
        )

    @patch.object(GENERATE_MUSIC_MODULE, "apply_repaint_waveform_splice")
    def test_aggressive_repaint_with_range_still_splices_outside_region(self, splice_mock):
        """Non-lyric aggressive repaint preserves source waveform outside the range."""
        host = _Host()
        spliced = torch.full((1, 2, 8), 9.0)
        splice_mock.return_value = spliced

        def _prepare_service_inputs(**kwargs):
            """Return bounded repaint service inputs for aggressive mode."""
            host.calls["_prepare_generate_music_service_inputs"] = kwargs
            return {
                "should_return_intermediate": True,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [10.0],
                "target_wavs_tensor": torch.ones(1, 2, 8),
            }

        host._prepare_generate_music_service_inputs = _prepare_service_inputs

        out = host.generate_music(
            captions="cap",
            lyrics="",
            task_type="repaint",
            src_audio="source.wav",
            repainting_start=0.0,
            repainting_end=10.0,
            repaint_mode="aggressive",
            repaint_strength=1.0,
        )

        self.assertEqual(out, host._final_payload)
        splice_mock.assert_called_once()
        self.assertIs(
            spliced,
            host.calls["_build_generate_music_success_payload"]["pred_wavs"],
        )

    @patch.object(GENERATE_MUSIC_MODULE, "apply_repaint_waveform_splice")
    @patch.object(GENERATE_MUSIC_MODULE, "apply_repaint_segment_splice")
    def test_lyric_repaint_generates_local_span_then_splices_into_source(
        self,
        segment_splice_mock,
        waveform_splice_mock,
    ):
        """Lyric repaint should generate only the selected span before insertion."""
        host = _Host()
        source_audio = torch.ones(2, 48000 * 3)
        spliced = torch.full((1, 2, 48000 * 3), 5.0)
        segment_splice_mock.return_value = spliced

        def _prepare_audio(**kwargs):
            """Return a three-second prepared source audio tensor."""
            host.calls["_prepare_reference_and_source_audio"] = kwargs
            return [[torch.zeros(2, 10)]], source_audio, None

        def _prepare_service_inputs(**kwargs):
            """Return service inputs for a one-second local repaint generation."""
            host.calls["_prepare_generate_music_service_inputs"] = kwargs
            return {
                "should_return_intermediate": True,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [1.0],
                "target_wavs_tensor": torch.zeros(1, 2, 48000),
            }

        host._prepare_reference_and_source_audio = _prepare_audio
        host._prepare_generate_music_service_inputs = _prepare_service_inputs

        out = host.generate_music(
            captions="rap",
            lyrics="new lyric line",
            task_type="repaint",
            src_audio="source.wav",
            audio_duration=99.0,
            repainting_start=1.0,
            repainting_end=2.0,
            repaint_mode="balanced",
            repaint_strength=0.5,
        )

        self.assertEqual(out, host._final_payload)
        self.assertIsNone(
            host.calls["_prepare_generate_music_service_inputs"]["processed_src_audio"],
        )
        self.assertEqual(
            1.0,
            host.calls["_prepare_generate_music_service_inputs"]["audio_duration"],
        )
        self.assertEqual(
            0.0,
            host.calls["_prepare_generate_music_service_inputs"]["repainting_start"],
        )
        self.assertEqual(
            1.0,
            host.calls["_prepare_generate_music_service_inputs"]["repainting_end"],
        )
        self.assertEqual(
            "text2music",
            host.calls["_prepare_generate_music_service_inputs"]["task_type"],
        )
        self.assertIn(
            "clear English vocal",
            host.calls["_prepare_generate_music_service_inputs"]["captions"],
        )
        self.assertEqual(
            1.0,
            host.calls["_run_generate_music_service_with_progress"]["audio_duration"],
        )
        self.assertEqual(
            "text2music",
            host.calls["_run_generate_music_service_with_progress"]["task_type"],
        )
        self.assertEqual(
            0.0,
            host.calls["_run_generate_music_service_with_progress"]["repaint_injection_ratio"],
        )
        waveform_splice_mock.assert_not_called()
        segment_splice_mock.assert_called_once()
        self.assertIs(
            source_audio,
            segment_splice_mock.call_args.kwargs["src_wavs"],
        )
        self.assertEqual(
            [1.0],
            segment_splice_mock.call_args.kwargs["repainting_starts"],
        )
        self.assertEqual(
            [2.0],
            segment_splice_mock.call_args.kwargs["repainting_ends"],
        )
        self.assertIs(
            spliced,
            host.calls["_build_generate_music_success_payload"]["pred_wavs"],
        )


class VramPreflightCheckTests(unittest.TestCase):
    """Verify ``_vram_preflight_check`` respects CPU offload mode."""

    _GM_MOD = GENERATE_MUSIC_MODULE

    @patch.object(_GM_MOD, "get_effective_free_vram_gb", return_value=5.5)
    @patch.object(_GM_MOD, "torch")
    def test_preflight_blocks_full_dit_offload_when_vram_low(
        self, mock_torch, _mock_free_vram
    ):
        """It includes the temporary DiT transfer footprint for full offload."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(
            offload_to_cpu=True,
            offload_dit_to_cpu=True,
            quantization="int8_weight_only",
            is_turbo=True,
        )
        result = host._vram_preflight_check(
            actual_batch_size=1,
            audio_duration=10.0,
            guidance_scale=1.0,
        )
        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertIn("Insufficient free VRAM", result["error"])

    @patch.object(_GM_MOD, "get_effective_free_vram_gb", return_value=7.5)
    @patch.object(_GM_MOD, "torch")
    def test_preflight_passes_full_dit_offload_when_vram_sufficient(
        self, mock_torch, _mock_free_vram
    ):
        """It allows the observed 8GB XL full-offload default path."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(
            offload_to_cpu=True,
            offload_dit_to_cpu=True,
            quantization="int8_weight_only",
            is_turbo=True,
        )
        result = host._vram_preflight_check(
            actual_batch_size=1,
            audio_duration=10.0,
            guidance_scale=1.0,
        )
        self.assertIsNone(result)

    @patch.object(_GM_MOD, "get_effective_free_vram_gb", return_value=3.4)
    @patch.object(_GM_MOD, "torch")
    def test_preflight_blocks_when_offload_disabled_and_vram_low(
        self, mock_torch, _mock_free_vram
    ):
        """It returns error payload when offload is off and free VRAM is insufficient."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(offload_to_cpu=False)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertIn("Insufficient free VRAM", result["error"])

    @patch.object(_GM_MOD, "get_effective_free_vram_gb", return_value=24.0)
    @patch.object(_GM_MOD, "torch")
    def test_preflight_passes_when_offload_disabled_and_vram_sufficient(
        self, mock_torch, _mock_free_vram
    ):
        """It returns None when offload is off but free VRAM exceeds estimate."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(offload_to_cpu=False)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNone(result)

    @patch.object(_GM_MOD, "torch")
    def test_preflight_passes_on_non_cuda_device(self, mock_torch):
        """It returns None when CUDA is not available (CPU/MPS/XPU)."""
        mock_torch.cuda.is_available.return_value = False
        host = _Host(offload_to_cpu=False)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNone(result)


class TurboGuidanceScaleTests(unittest.TestCase):
    """Verify turbo models force guidance_scale to 1.0 (issue #927)."""

    def test_turbo_model_overrides_guidance_scale_to_one(self):
        """Turbo model should clamp guidance_scale to 1.0 before service call."""
        host = _Host(is_turbo=True)
        host._readiness_error = None
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            inference_steps=8,
            guidance_scale=7.0,
            use_random_seed=False,
            seed=77,
            task_type="text2music",
        )
        self.assertEqual(out, host._final_payload)
        self.assertEqual(
            host.calls["_run_generate_music_service_with_progress"]["guidance_scale"],
            1.0,
        )

    def test_non_turbo_model_preserves_guidance_scale(self):
        """Non-turbo model should keep the user-provided guidance_scale."""
        host = _Host(is_turbo=False)
        host._readiness_error = None
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            inference_steps=8,
            guidance_scale=7.0,
            use_random_seed=False,
            seed=77,
            task_type="text2music",
        )
        self.assertEqual(out, host._final_payload)
        self.assertEqual(
            host.calls["_run_generate_music_service_with_progress"]["guidance_scale"],
            7.0,
        )


if __name__ == "__main__":
    unittest.main()
