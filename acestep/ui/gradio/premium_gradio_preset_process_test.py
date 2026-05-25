"""Gradio process-api regression tests for premium preset loading."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from acestep.model_downloader import (
    DEFAULT_LM_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
)
from acestep.ui.gradio.premium_app import create_gradio_interface
from acestep.ui.gradio.premium_features import (
    DEFAULT_PRESET_VALUES,
    get_preset_component_keys,
    save_preset_action,
)
from acestep.ui.gradio.premium_preset_schema import FILE_UPLOAD_PRESET_KEYS


class _FakeDitHandler:
    """Minimal DiT handler for constructing the premium Gradio app."""

    model = None

    def get_available_acestep_v15_models(self) -> list[str]:
        """Return model choices used by app construction."""

        return [DEFAULT_PREMIUM_DIT_MODEL, DEFAULT_TURBO_DIT_MODEL]

    def get_available_checkpoints(self) -> list[str]:
        """Return checkpoint choices used by app construction."""

        return [str(Path(os.environ["ACESTEP_PROJECT_ROOT"]) / "models")]

    def is_flash_attention_available(self, device: str) -> bool:
        """Return a stable flash-attention capability for UI construction."""

        return True

    def is_turbo_model(self) -> bool:
        """Return whether the fake initialized model should use turbo defaults."""

        return True


class _FakeLlmHandler:
    """Minimal LM handler for constructing the premium Gradio app."""

    use_legacy_cfg_prompt = False

    def get_available_5hz_lm_models(self) -> list[str]:
        """Return LM choices used by app construction."""

        return [DEFAULT_LM_MODEL, "acestep-5Hz-lm-4B"]


class _FakeDatasetHandler:
    """Minimal dataset handler for Dataset page event registration."""

    dataset_imported = False

    def import_dataset_for_ui(self, dataset_type: str, dataset_path: str) -> tuple[Any, ...]:
        """Return empty dataset import outputs."""

        return ("No dataset imported", None, None, None, None, None)

    def get_item_for_ui(self, search_type: str, search_value: str) -> tuple[Any, ...]:
        """Return empty dataset item outputs."""

        return ("No dataset imported", None, None, None, None, None)


class PremiumGradioPresetProcessTests(unittest.IsolatedAsyncioTestCase):
    """Verify preset callbacks survive Gradio component postprocessing."""

    async def test_load_preset_process_api_accepts_empty_upload_values(self) -> None:
        """Loading a preset should not make Gradio cache empty upload paths."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
                for key in FILE_UPLOAD_PRESET_KEYS:
                    values[keys.index(key)] = ""
                save_preset_action("api-load", None, *values)

                demo = create_gradio_interface(
                    dit_handler=_FakeDitHandler(),
                    llm_handler=_FakeLlmHandler(),
                    dataset_handler=_FakeDatasetHandler(),
                    init_params=None,
                    language="en",
                )
                load_id = next(
                    key for key, block_fn in demo.fns.items()
                    if getattr(block_fn.fn, "__name__", "") == "load_preset_action"
                )
                demo.fns[load_id].inputs[0].choices = [("api-load", "api-load")]

                result = await demo.process_api(load_id, ["api-load"])
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(len(result.get("data", [])), len(keys) + 5)


if __name__ == "__main__":
    unittest.main()
