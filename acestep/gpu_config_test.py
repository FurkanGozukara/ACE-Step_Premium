"""Unit tests for GPU-config LM backend compatibility helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.gpu_config import (
    GPU_TIER_LABELS,
    get_default_quantization_method,
    get_gpu_config,
    get_gpu_config_for_tier,
    get_gpu_tier,
    resolve_lm_backend,
)


class GpuConfigLegacyCudaTests(unittest.TestCase):
    """Verify legacy CUDA devices steer the LM backend away from vLLM."""

    def test_get_gpu_config_forces_pt_backend_on_legacy_cuda(self) -> None:
        """Pre-Volta CUDA devices should expose a PyTorch-only LM recommendation."""
        with patch("acestep.gpu_config.is_legacy_cuda_gpu", return_value=True):
            config = get_gpu_config(gpu_memory_gb=12.0)

        self.assertEqual("pt", config.recommended_backend)
        self.assertEqual("pt_only", config.lm_backend_restriction)

    def test_resolve_lm_backend_forces_pt_when_gpu_is_legacy(self) -> None:
        """vLLM requests should collapse to PyTorch on legacy CUDA GPUs."""
        config = SimpleNamespace(recommended_backend="pt", lm_backend_restriction="pt_only")
        self.assertEqual("pt", resolve_lm_backend("vllm", config))
        self.assertEqual("pt", resolve_lm_backend(None, config))

    def test_resolve_lm_backend_keeps_vllm_when_hardware_allows_it(self) -> None:
        """Modern CUDA tiers should keep the requested vLLM backend."""
        config = SimpleNamespace(recommended_backend="vllm", lm_backend_restriction="all")
        self.assertEqual("vllm", resolve_lm_backend("vllm", config))

    def test_get_gpu_config_prefers_pt_when_vllm_preflight_warns(self) -> None:
        """CUDA setups without vLLM prerequisites should default to PyTorch LM."""
        with patch("acestep.gpu_config.is_legacy_cuda_gpu", return_value=False), patch(
            "acestep.gpu_config.is_cuda_available",
            return_value=True,
        ), patch(
            "acestep.llm_backend_compat.get_vllm_preflight_warning",
            return_value="vLLM unavailable",
        ):
            config = get_gpu_config(gpu_memory_gb=32.0)

        self.assertEqual("pt", config.recommended_backend)


class GpuConfigMeasuredPresetTests(unittest.TestCase):
    """Verify measured XL preset defaults stay conservative by VRAM tier."""

    def test_sub_6gb_auto_selects_cpu_tier(self) -> None:
        """GPUs below the tier3 floor should auto-select the CPU tier."""
        config = get_gpu_config(gpu_memory_gb=5.49)

        self.assertEqual("tier1", config.tier)
        self.assertFalse(config.init_lm_default)
        self.assertEqual("pt", config.recommended_backend)
        self.assertEqual("pt_only", config.lm_backend_restriction)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertTrue(config.offload_dit_to_cpu_default)
        self.assertFalse(config.quantization_default)
        self.assertFalse(config.compile_model_default)

    def test_tier3_presets_low_vram_no_lm_int8_full_offload(self) -> None:
        """6-8GB GPUs use the lowest measured GPU path."""
        config = get_gpu_config(gpu_memory_gb=6.5)

        self.assertEqual("tier3", config.tier)
        self.assertFalse(config.init_lm_default)
        self.assertEqual("acestep-5Hz-lm-0.6B", config.recommended_lm_model)
        self.assertEqual(1, config.max_batch_size_with_lm)
        self.assertEqual(1, config.max_batch_size_without_lm)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertTrue(config.offload_dit_to_cpu_default)
        self.assertTrue(config.quantization_default)
        self.assertEqual("int8_weight_only", get_default_quantization_method(config))
        self.assertFalse(config.generate_lm_audio_codes_default)
        self.assertFalse(config.dcw_enabled_default)
        self.assertFalse(config.compile_model_default)

    def test_tier4_presets_full_offload_at_batch_one(self) -> None:
        """8-12GB GPUs use the conservative measured 0.6B LM path."""
        config = get_gpu_config(gpu_memory_gb=10.5)

        self.assertEqual("tier4", config.tier)
        self.assertTrue(config.init_lm_default)
        self.assertEqual(1, config.max_batch_size_with_lm)
        self.assertEqual(1, config.max_batch_size_without_lm)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertTrue(config.offload_dit_to_cpu_default)
        self.assertTrue(config.quantization_default)
        self.assertFalse(config.compile_model_default)

    def test_tier5_presets_batch_one_under_int8_offload(self) -> None:
        """12-16GB GPUs use the measured INT8 path while keeping batch-1 presets."""
        config = get_gpu_config(gpu_memory_gb=13.0)

        self.assertEqual("tier5", config.tier)
        self.assertEqual(1, config.max_batch_size_with_lm)
        self.assertEqual(1, config.max_batch_size_without_lm)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertFalse(config.offload_dit_to_cpu_default)
        self.assertTrue(config.quantization_default)
        self.assertFalse(config.compile_model_default)

    def test_tier6a_recommends_4b_with_int8_offload_at_batch_one(self) -> None:
        """16-24GB GPUs use the measured batch-1 4B LM path."""
        config = get_gpu_config(gpu_memory_gb=16.0)

        self.assertEqual("tier6a", config.tier)
        self.assertEqual(1, config.max_batch_size_with_lm)
        self.assertEqual(1, config.max_batch_size_without_lm)
        self.assertIn("acestep-5Hz-lm-4B", config.available_lm_models)
        self.assertEqual("acestep-5Hz-lm-4B", config.recommended_lm_model)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertFalse(config.offload_dit_to_cpu_default)
        self.assertTrue(config.quantization_default)
        self.assertFalse(config.compile_model_default)

    def test_tier6a_extends_until_24gb_tolerance_floor(self) -> None:
        """GPUs below the 24GB tolerance floor remain tier6a."""
        config = get_gpu_config(gpu_memory_gb=23.49)

        self.assertEqual("tier6a", config.tier)
        self.assertEqual("acestep-5Hz-lm-4B", config.recommended_lm_model)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertTrue(config.quantization_default)

    def test_tier6b_remains_manual_safe_profile(self) -> None:
        """The 24GB safe profile remains selectable manually."""
        with patch("acestep.gpu_config.get_gpu_memory_gb", return_value=24.0):
            config = get_gpu_config_for_tier("tier6b")

        self.assertEqual("tier6b", config.tier)
        self.assertEqual(1, config.max_batch_size_with_lm)
        self.assertEqual(1, config.max_batch_size_without_lm)
        self.assertIn("acestep-5Hz-lm-4B", config.available_lm_models)
        self.assertEqual("acestep-5Hz-lm-4B", config.recommended_lm_model)
        self.assertTrue(config.offload_to_cpu_default)
        self.assertFalse(config.offload_dit_to_cpu_default)
        self.assertFalse(config.quantization_default)
        self.assertFalse(config.compile_model_default)

    def test_unlimited_recommends_measured_4b_at_batch_one(self) -> None:
        """24GB-class and larger GPUs auto-select unlimited."""
        config = get_gpu_config(gpu_memory_gb=23.9995)

        self.assertEqual("unlimited", config.tier)
        self.assertEqual(1, config.max_batch_size_with_lm)
        self.assertEqual(1, config.max_batch_size_without_lm)
        self.assertEqual("acestep-5Hz-lm-4B", config.recommended_lm_model)
        self.assertFalse(config.offload_to_cpu_default)
        self.assertFalse(config.compile_model_default)

    def test_tier_boundaries_use_half_gb_tolerance(self) -> None:
        """Advertised VRAM classes tolerate reports up to 0.5GB below nominal."""
        cases = {
            0.0: "tier1",
            5.49: "tier1",
            5.5: "tier3",
            7.49: "tier3",
            7.5: "tier4",
            11.49: "tier4",
            11.5: "tier5",
            15.49: "tier5",
            15.5: "tier6a",
            23.49: "tier6a",
            23.5: "unlimited",
            23.9995: "unlimited",
            31.8423: "unlimited",
        }
        for memory_gb, expected_tier in cases.items():
            with self.subTest(memory_gb=memory_gb):
                self.assertEqual(expected_tier, get_gpu_tier(memory_gb))

    def test_removed_tier2_is_not_ui_choice(self) -> None:
        """Removed GPU tier2 should not be exposed through UI labels."""
        self.assertNotIn("tier2", GPU_TIER_LABELS)
        self.assertEqual("tier1 (CPU)", GPU_TIER_LABELS["tier1"])
        self.assertEqual("tier3 (6-8GB / peak 6.6GiB)", GPU_TIER_LABELS["tier3"])


class AutoMlxVaeChunkSizeTests(unittest.TestCase):
    """Tests for memory-based MLX VAE chunk size selection."""

    def test_low_memory_returns_256(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=16), 256)

    def test_mid_memory_returns_512(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=36), 512)

    def test_high_memory_returns_1024(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=64), 1024)

    def test_very_high_memory_returns_2048(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=128), 2048)

    def test_env_var_override(self):
        import os
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE_CHUNK": "1024"}):
            self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=16), 1024)

    def test_env_var_clamps_to_minimum(self):
        import os
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE_CHUNK": "32"}):
            self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=16), 192)

    def test_invalid_env_var_falls_back_to_memory(self):
        import os
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE_CHUNK": "not_a_number"}):
            self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=16), 256)

    def test_boundary_17gb(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=17), 512)

    def test_boundary_37gb(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=37), 1024)

    def test_boundary_65gb(self):
        from acestep.gpu_config import _auto_mlx_vae_chunk_size
        self.assertEqual(_auto_mlx_vae_chunk_size(mem_gb=65), 2048)


if __name__ == "__main__":
    unittest.main()
