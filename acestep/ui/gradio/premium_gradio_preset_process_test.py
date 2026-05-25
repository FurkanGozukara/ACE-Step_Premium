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
    list_preset_names,
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

    def _create_demo_with_presets(
        self,
        preset_captions: dict[str, str],
    ) -> tuple[Any, tuple[str, ...], int]:
        """Create the real premium Gradio app with saved test presets."""

        keys = get_preset_component_keys()
        for preset_name, caption in preset_captions.items():
            values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
            values[keys.index("captions")] = caption
            save_preset_action(preset_name, None, *values)

        demo = create_gradio_interface(
            dit_handler=_FakeDitHandler(),
            llm_handler=_FakeLlmHandler(),
            dataset_handler=_FakeDatasetHandler(),
            init_params=None,
            language="en",
        )
        delete_id = next(
            key for key, block_fn in demo.fns.items()
            if getattr(block_fn.fn, "__name__", "") == "delete_preset_action"
        )
        names = list_preset_names()
        demo.fns[delete_id].inputs[0].choices = [(name, name) for name in names]
        return demo, keys, delete_id

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

    async def test_load_preset_process_api_sanitizes_strict_control_values(self) -> None:
        """Loading malformed saved values should keep later Gradio events valid."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
                values[keys.index("train_epochs")] = 100200
                values[keys.index("lokr_export_epoch")] = "Latest (Auto)"
                save_preset_action("malformed", None, *values)

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
                save_id = next(
                    key for key, block_fn in demo.fns.items()
                    if getattr(block_fn.fn, "__name__", "") == "save_preset_action"
                )
                demo.fns[load_id].inputs[0].choices = [("malformed", "malformed")]
                demo.fns[save_id].inputs[1].choices = [("malformed", "malformed")]

                load_result = await demo.process_api(load_id, ["malformed"])
                loaded_values = [
                    item.get("value") if isinstance(item, dict) else item
                    for item in load_result.get("data", [])[:len(keys)]
                ]
                save_result = await demo.process_api(
                    save_id,
                    ["sanitized-copy", "malformed", *loaded_values],
                )
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        data = load_result.get("data", [])
        self.assertEqual(data[keys.index("train_epochs")].get("value"), 4000)
        self.assertEqual(
            data[keys.index("lokr_export_epoch")].get("value"),
            "Latest (auto)",
        )
        self.assertIn("Saved preset: sanitized-copy", save_result.get("data", [])[1])

    async def test_delete_preset_process_api_selects_next_available_preset(self) -> None:
        """Deleting the selected preset should refresh and load the next preset."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                demo, keys, delete_id = self._create_demo_with_presets(
                    {"alpha": "alpha caption", "beta": "beta caption"}
                )
                delete_button = next(
                    block for block in demo.blocks.values()
                    if getattr(block, "value", None) == "Delete Preset"
                )
                self.assertIn("action-btn-delete-preset", delete_button.elem_classes)
                self.assertNotIn("action-btn-cancel", delete_button.elem_classes)

                default_state = demo.fns[delete_id].inputs[1]
                result = await demo.process_api(
                    delete_id,
                    ["alpha", default_state.value, ""],
                )
                remaining = list_preset_names()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        data = result.get("data", [])
        self.assertEqual(len(data), len(keys) + 5)
        self.assertEqual(["beta"], remaining)
        self.assertEqual("beta caption", data[keys.index("captions")].get("value"))
        self.assertEqual("beta", data[len(keys) + 2].get("value"))
        self.assertIn("Deleted preset: alpha", data[len(keys) + 3])

    async def test_delete_last_preset_process_api_restores_defaults(self) -> None:
        """Deleting the final preset should clear choices and restore defaults."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                demo, keys, delete_id = self._create_demo_with_presets(
                    {"only": "last preset caption"}
                )
                default_state = demo.fns[delete_id].inputs[1]
                result = await demo.process_api(
                    delete_id,
                    ["only", default_state.value, ""],
                )
                remaining = list_preset_names()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        data = result.get("data", [])
        dropdown_update = data[len(keys) + 2]
        self.assertEqual(len(data), len(keys) + 5)
        self.assertEqual([], remaining)
        self.assertEqual([], dropdown_update.get("choices"))
        self.assertIsNone(dropdown_update.get("value"))
        self.assertNotEqual(
            "last preset caption",
            data[keys.index("captions")].get("value"),
        )
        self.assertIn("Using GPU Optimization Preset", data[len(keys) + 3])


if __name__ == "__main__":
    unittest.main()
