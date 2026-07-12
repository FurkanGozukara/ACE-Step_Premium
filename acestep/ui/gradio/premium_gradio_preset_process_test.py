"""Gradio process-api regression tests for premium preset loading."""

from __future__ import annotations

import os
import tempfile
import unittest
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from gradio.state_holder import SessionState

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

    async def test_load_preset_process_api_migrates_removed_vram_tiers(self) -> None:
        """Removed VRAM tier values should load as the detected GPU tier."""

        fake_gpu_config = SimpleNamespace(
            tier="unlimited",
            mlx_vae_chunk_size=1024,
            recommended_lm_model="acestep-5Hz-lm-4B",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
                values[keys.index("tier_dropdown")] = "tier2"
                values[keys.index("simple_create_tier_dropdown")] = "tier3"
                with patch(
                    "acestep.ui.gradio.premium_features.get_global_gpu_config",
                    return_value=fake_gpu_config,
                ):
                    save_preset_action("legacy-tier", None, *values)
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
                    demo.fns[load_id].inputs[0].choices = [
                        ("legacy-tier", "legacy-tier")
                    ]
                    result = await demo.process_api(load_id, ["legacy-tier"])
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        data = result.get("data", [])
        self.assertEqual(data[keys.index("tier_dropdown")].get("value"), "unlimited")
        self.assertEqual(
            data[keys.index("simple_create_tier_dropdown")].get("value"),
            "unlimited",
        )

    async def test_load_preset_process_api_restores_sam_prompt_multiselect(self) -> None:
        """SAM Quick Prompt multiselect values should load through real components."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
                values[keys.index("sam_prompt_preset")] = ["vocals", "bass"]
                values[keys.index("sam_batch_segment")] = True
                save_preset_action("sam-prompts", None, *values)

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
                demo.fns[load_id].inputs[0].choices = [("sam-prompts", "sam-prompts")]

                result = await demo.process_api(load_id, ["sam-prompts"])
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        data = result.get("data", [])
        self.assertEqual(
            ["vocals", "bass"],
            data[keys.index("sam_prompt_preset")].get("value"),
        )
        self.assertTrue(data[keys.index("sam_batch_segment")].get("value"))

    async def test_load_preset_process_api_restores_complete_track_checkbox_group(
        self,
    ) -> None:
        """Complete track lists, including empty lists, must survive real components."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                keys = get_preset_component_keys()
                values = [DEFAULT_PRESET_VALUES.get(key, "") for key in keys]
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

                values[keys.index("complete_track_classes")] = ["drums", "vocals"]
                save_preset_action("complete-tracks", None, *values)
                demo.fns[load_id].inputs[0].choices = [
                    ("complete-tracks", "complete-tracks")
                ]
                selected_result = await demo.process_api(
                    load_id,
                    ["complete-tracks"],
                )

                values[keys.index("complete_track_classes")] = []
                save_preset_action("complete-empty", None, *values)
                demo.fns[load_id].inputs[0].choices = [
                    ("complete-empty", "complete-empty")
                ]
                empty_result = await demo.process_api(load_id, ["complete-empty"])
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        index = keys.index("complete_track_classes")
        self.assertEqual(
            ["drums", "vocals"],
            selected_result["data"][index].get("value"),
        )
        self.assertEqual([], empty_result["data"][index].get("value"))

    async def test_every_preset_component_round_trips_through_process_api(
        self,
    ) -> None:
        """Exercise every tracked GUI value through Gradio save and load callbacks."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                keys = get_preset_component_keys()
                demo = create_gradio_interface(
                    dit_handler=_FakeDitHandler(),
                    llm_handler=_FakeLlmHandler(),
                    dataset_handler=_FakeDatasetHandler(),
                    init_params=None,
                    language="en",
                )
                save_id = next(
                    key for key, block_fn in demo.fns.items()
                    if getattr(block_fn.fn, "__name__", "") == "save_preset_action"
                )
                load_id = next(
                    key for key, block_fn in demo.fns.items()
                    if getattr(block_fn.fn, "__name__", "") == "load_preset_action"
                )
                components = demo.fns[save_id].inputs[2:]
                self.assertEqual(len(keys), len(components))
                values = [
                    _alternate_component_value(key, component)
                    for key, component in zip(keys, components)
                ]
                _align_cross_tab_preset_values(keys, values)

                save_result = await demo.process_api(
                    save_id,
                    ["all-components", None, *values],
                )
                demo.fns[load_id].inputs[0].choices = [
                    ("all-components", "all-components")
                ]
                load_state = SessionState(demo)
                load_result = await demo.process_api(
                    load_id,
                    ["all-components"],
                    state=load_state,
                    session_hash="all-components-load",
                )
                loaded_values = [
                    item.get("value") if isinstance(item, dict) else item
                    for item in load_result["data"][:len(keys)]
                ]
                for index, component in enumerate(components):
                    if type(component).__name__ == "State":
                        output_component = demo.fns[load_id].outputs[index]
                        loaded_values[index] = load_state[output_component._id]
                demo.fns[save_id].inputs[1].choices = [
                    ("all-components", "all-components")
                ]
                copy_save_result = await demo.process_api(
                    save_id,
                    [
                        "all-components-copy",
                        "all-components",
                        *loaded_values,
                    ],
                )
                demo.fns[load_id].inputs[0].choices = [
                    ("all-components-copy", "all-components-copy")
                ]
                copy_load_state = SessionState(demo)
                copy_load_result = await demo.process_api(
                    load_id,
                    ["all-components-copy"],
                    state=copy_load_state,
                    session_hash="all-components-copy-load",
                )
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertIn("Saved preset: all-components", save_result["data"][1])
        self.assertIn(
            "Saved preset: all-components-copy",
            copy_save_result["data"][1],
        )
        self.assertEqual(len(keys) + 5, len(load_result["data"]))
        self.assertEqual(len(keys) + 5, len(copy_load_result["data"]))
        for index, key in enumerate(keys):
            actual_update = load_result["data"][index]
            copied_update = copy_load_result["data"][index]
            if type(components[index]).__name__ == "State":
                actual_value = load_state[demo.fns[load_id].outputs[index]._id]
                copied_value = copy_load_state[
                    demo.fns[load_id].outputs[index]._id
                ]
            else:
                self.assertIsInstance(actual_update, dict, key)
                self.assertIsInstance(copied_update, dict, key)
                actual_value = _process_output_value(actual_update)
                copied_value = _process_output_value(copied_update)
            self.assertEqual(values[index], actual_value, key)
            self.assertEqual(actual_value, copied_value, key)

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


def _choice_value(choice: Any) -> Any:
    """Return the submitted value from a Gradio choice definition."""

    if isinstance(choice, (list, tuple)) and len(choice) >= 2:
        return choice[1]
    return choice


def _process_output_value(value: Any) -> Any:
    """Return a value from either a Gradio update or a raw State output."""

    if isinstance(value, dict):
        return value.get("value")
    return value


def _alternate_component_value(key: str, component: Any) -> Any:
    """Return a different but component-valid value for exhaustive round trips."""

    component_type = type(component).__name__
    current = getattr(component, "value", None)
    if component_type == "Checkbox":
        return not bool(current)
    if component_type in {"File", "Image", "Audio"}:
        return None

    choices = [
        _choice_value(choice)
        for choice in (getattr(component, "choices", None) or [])
    ]
    is_multiselect = bool(getattr(component, "multiselect", False))
    if component_type == "CheckboxGroup" or is_multiselect:
        return choices[-2:] if len(choices) >= 2 else choices
    if choices:
        return next((choice for choice in reversed(choices) if choice != current), choices[0])

    if component_type in {"Slider", "Number"}:
        candidates = [
            getattr(component, "maximum", None),
            getattr(component, "minimum", None),
        ]
        for candidate in candidates:
            if (
                isinstance(candidate, (int, float))
                and not isinstance(candidate, bool)
                and isfinite(float(candidate))
                and candidate != current
            ):
                return candidate
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return current + 1
        return 1

    if component_type == "Textbox":
        return f"codex-{key}"
    return current


def _align_cross_tab_preset_values(keys: tuple[str, ...], values: list[Any]) -> None:
    """Give synchronized controls one coherent alternate value for fidelity checks."""

    def set_value(key: str, value: Any) -> None:
        values[keys.index(key)] = value

    set_value("generation_mode", "Remix")
    set_value("task_type", "cover")
    set_value("config_path", DEFAULT_PREMIUM_DIT_MODEL)
    set_value("simple_model_dropdown", DEFAULT_PREMIUM_DIT_MODEL)
    for key in (
        "vocal_language",
        "simple_vocal_language",
        "simple_create_vocal_language",
    ):
        set_value(key, "tr")
    for key in ("lm_negative_prompt", "simple_create_negative_prompt"):
        set_value(key, "same negative prompt")
    for key in ("sampler_mode", "simple_create_sampler_mode"):
        set_value(key, "euler")
    set_value(
        "tier_dropdown",
        values[keys.index("simple_create_tier_dropdown")],
    )


if __name__ == "__main__":
    unittest.main()
