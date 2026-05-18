"""Unit tests for CLI startup LM backend resolution in the pipeline."""

from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep import acestep_v15_pipeline


class PipelineStartupBackendTests(unittest.TestCase):
    """Verify startup backend resolution respects legacy CUDA restrictions."""

    def _legacy_gpu_config(self) -> SimpleNamespace:
        """Return a representative legacy-CUDA GPU configuration."""
        return SimpleNamespace(
            gpu_memory_gb=12.0,
            tier="tier5",
            max_duration_with_lm=480,
            max_duration_without_lm=600,
            max_batch_size_with_lm=1,
            max_batch_size_without_lm=1,
            init_lm_default=True,
            available_lm_models=["acestep-5Hz-lm-0.6B"],
            recommended_lm_model="acestep-5Hz-lm-0.6B",
            recommended_backend="pt",
            lm_backend_restriction="pt_only",
            offload_dit_to_cpu_default=False,
            quantization_default=False,
        )

    def _run_main(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        gpu_config: SimpleNamespace | None = None,
        available_dit_models: list[str] | None = None,
        available_lm_models: list[str] | None = None,
    ) -> tuple[MagicMock, dict[str, object]]:
        """Run ``main`` with heavy dependencies stubbed and capture startup state."""
        gpu_config = gpu_config or self._legacy_gpu_config()
        dit_handler = MagicMock()
        dit_handler.get_available_acestep_v15_models.return_value = (
            available_dit_models or ["acestep-v15-turbo"]
        )
        dit_handler.is_flash_attention_available.return_value = False
        dit_handler.initialize_service.return_value = ("ok", True)

        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = (
            available_lm_models or ["acestep-5Hz-lm-0.6B"]
        )
        llm_handler.initialize.return_value = ("ok", True)

        demo = MagicMock()
        demo.app = MagicMock(name="queued_gradio_app")
        demo.queue.return_value = demo
        demo.launch.return_value = None

        captured: dict[str, object] = {}
        captured["demo"] = demo
        captured["dit_handler"] = dit_handler
        startup_gpu_state = SimpleNamespace(
            gpu_config=gpu_config,
            device_name="Test CUDA GPU",
            device_kind="cuda",
            cuda_compute_major=9,
        )
        ace_step_handler_cls = MagicMock(return_value=dit_handler)
        llm_handler_cls = MagicMock(return_value=llm_handler)

        def _create_demo(init_params=None, language="en"):
            captured["init_params"] = init_params
            captured["language"] = language
            return demo

        def _register_cancel_route(route_demo):
            captured["cancel_route_registered_demo"] = route_demo
            captured["cancel_route_registered_after_queue"] = demo.queue.called
            captured["cancel_route_registered_app"] = route_demo.app

        with patch.object(sys, "argv", argv), patch.dict(os.environ, env or {}, clear=True), patch(
            "acestep.acestep_v15_pipeline.build_startup_gpu_state",
            return_value=startup_gpu_state,
        ), patch(
            "acestep.acestep_v15_pipeline.set_global_gpu_config"
        ), patch(
            "acestep.acestep_v15_pipeline.get_i18n"
        ), patch(
            "acestep.acestep_v15_pipeline.available_languages_info",
            return_value=[("en", "English", "English")],
        ), patch(
            "acestep.acestep_v15_pipeline._import_real_handlers",
            return_value=(ace_step_handler_cls, llm_handler_cls),
        ), patch(
            "acestep.acestep_v15_pipeline.create_demo",
            side_effect=_create_demo,
        ), patch(
            "acestep.acestep_v15_pipeline._import_generation_cancel_route",
            return_value=_register_cancel_route,
        ), patch(
            "acestep.acestep_v15_pipeline.ensure_lm_model",
            return_value=(True, "ok"),
        ), patch(
            "acestep.acestep_v15_pipeline.os.makedirs"
        ):
            acestep_v15_pipeline.main()

        return llm_handler, captured

    def test_main_registers_cancel_route_after_queue_before_launch(self) -> None:
        """Queued Gradio startup should serve the out-of-band cancel route."""

        _llm_handler, captured = self._run_main(["acestep"])

        demo = captured["demo"]
        self.assertIs(captured["cancel_route_registered_demo"], demo)
        self.assertTrue(captured["cancel_route_registered_after_queue"])
        self.assertIs(
            demo.launch.call_args.kwargs["_app"],
            captured["cancel_route_registered_app"],
        )

    def test_main_forces_pt_backend_for_explicit_vllm_argument(self) -> None:
        """Legacy CUDA startup should override an explicit CLI vLLM request."""
        llm_handler, captured = self._run_main(
            [
                "acestep",
                "--init_service",
                "true",
                "--init_llm",
                "true",
                "--config_path",
                "acestep-v15-turbo",
                "--lm_model_path",
                "acestep-5Hz-lm-0.6B",
                "--backend",
                "vllm",
            ]
        )

        self.assertEqual("pt", llm_handler.initialize.call_args.kwargs["backend"])
        self.assertEqual("pt", captured["init_params"]["backend"])

    def test_main_auto_selects_recommended_4b_and_enables_offload(self) -> None:
        """Startup init should use the tier recommendation before DiT initialization."""
        gpu_config = SimpleNamespace(
            gpu_memory_gb=24.0,
            tier="tier6b",
            max_duration_with_lm=480,
            max_duration_without_lm=480,
            max_batch_size_with_lm=1,
            max_batch_size_without_lm=1,
            init_lm_default=True,
            available_lm_models=[
                "acestep-5Hz-lm-0.6B",
                "acestep-5Hz-lm-1.7B",
                "acestep-5Hz-lm-4B",
            ],
            recommended_lm_model="acestep-5Hz-lm-4B",
            recommended_backend="pt",
            lm_backend_restriction="all",
            offload_dit_to_cpu_default=False,
            quantization_default=False,
        )

        llm_handler, captured = self._run_main(
            [
                "acestep",
                "--init_service",
                "true",
                "--init_llm",
                "true",
                "--config_path",
                "acestep-v15-turbo",
                "--backend",
                "pt",
            ],
            gpu_config=gpu_config,
            available_lm_models=[
                "acestep-5Hz-lm-1.7B",
                "acestep-5Hz-lm-4B",
            ],
        )

        dit_handler = captured["dit_handler"]
        self.assertEqual(
            "acestep-5Hz-lm-4B",
            llm_handler.initialize.call_args.kwargs["lm_model_path"],
        )
        self.assertTrue(
            dit_handler.initialize_service.call_args.kwargs["offload_to_cpu"]
        )
        self.assertTrue(llm_handler.initialize.call_args.kwargs["offload_to_cpu"])
        self.assertTrue(captured["init_params"]["offload_to_cpu"])

    def test_main_forces_pt_backend_for_service_mode_backend_override(self) -> None:
        """Service mode should not re-enable vLLM on legacy CUDA hardware."""
        llm_handler, captured = self._run_main(
            [
                "acestep",
                "--service_mode",
                "true",
                "--init_service",
                "true",
                "--init_llm",
                "true",
            ],
            env={"SERVICE_MODE_BACKEND": "vllm"},
        )

        self.assertEqual("pt", llm_handler.initialize.call_args.kwargs["backend"])
        self.assertEqual("pt", captured["init_params"]["backend"])

    def test_main_auto_selects_xl_turbo_when_config_is_unset(self) -> None:
        """CLI service initialization should mirror the fresh Gradio default."""
        _llm_handler, captured = self._run_main(
            [
                "acestep",
                "--init_service",
                "true",
                "--init_llm",
                "false",
            ],
            available_dit_models=[
                "acestep-v15-xl-sft",
                "acestep-v15-xl-turbo",
            ],
        )

        dit_handler = captured["dit_handler"]
        self.assertEqual(
            "acestep-v15-xl-turbo",
            dit_handler.initialize_service.call_args.kwargs["config_path"],
        )
        self.assertEqual(
            "acestep-v15-xl-turbo",
            captured["init_params"]["config_path"],
        )

    def test_main_forces_pt_backend_for_api_env_override(self) -> None:
        """API-mode env overrides should still resolve to the safe startup backend."""
        api_routes_module = types.SimpleNamespace(setup_api_routes=MagicMock())

        with patch.dict(
            sys.modules,
            {"acestep.ui.gradio.api.api_routes": api_routes_module},
        ), patch("time.sleep", side_effect=KeyboardInterrupt):
            llm_handler, captured = self._run_main(
                [
                    "acestep",
                    "--enable-api",
                    "--init_llm",
                    "true",
                    "--config_path",
                    "acestep-v15-turbo",
                    "--lm_model_path",
                    "acestep-5Hz-lm-0.6B",
                ],
                env={"ACESTEP_LM_BACKEND": "vllm"},
            )

        self.assertEqual("pt", llm_handler.initialize.call_args.kwargs["backend"])
        self.assertEqual("pt", captured["init_params"]["backend"])


if __name__ == "__main__":
    unittest.main()
