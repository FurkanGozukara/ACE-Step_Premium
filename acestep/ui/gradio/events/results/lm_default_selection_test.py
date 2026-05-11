"""Tests for foreground LM default selection."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.events.results import batch_management_wrapper


class LmDefaultSelectionTests(unittest.TestCase):
    """Verify lazy foreground LM init follows GPU-tier recommendations."""

    def test_lm_auto_init_uses_gpu_recommended_model_when_unset(self) -> None:
        """Blank LM selection should resolve to the GPU-recommended local model."""
        llm_handler = MagicMock()
        llm_handler.llm_initialized = False
        llm_handler.get_available_5hz_lm_models.return_value = [
            "acestep-5Hz-lm-1.7B",
            "acestep-5Hz-lm-4B",
        ]
        gpu_config = SimpleNamespace(recommended_lm_model="acestep-5Hz-lm-4B")

        with patch.object(
            batch_management_wrapper,
            "get_global_gpu_config",
            return_value=gpu_config,
        ):
            needs_init, lm_model = batch_management_wrapper._lm_service_needs_init(
                llm_handler,
                init_llm_checkbox=True,
                think_checkbox=True,
                auto_score=False,
                lm_model_path="",
                backend_dropdown="pt",
                device="auto",
                offload_to_cpu=False,
            )

        self.assertTrue(needs_init)
        self.assertEqual(lm_model, "acestep-5Hz-lm-4B")


if __name__ == "__main__":
    unittest.main()
