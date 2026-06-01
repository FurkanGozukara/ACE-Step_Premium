"""Tests for user presets and GPU-optimization startup defaults."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
    get_models_dir,
)
from acestep.ui.gradio import premium_features


def _write_peft_adapter(path: Path) -> Path:
    """Create a minimal PEFT adapter directory."""

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"")
    return path


class PremiumFeaturesTests(unittest.TestCase):
    """Verify startup uses GPU defaults unless a user preset is loaded."""

    def _with_project_root(self):
        """Return a temporary install root context."""
        return tempfile.TemporaryDirectory()

    def _set_project_root(self, tmp_dir: str) -> str | None:
        """Set the test project root and return the previous value."""
        original = os.environ.get("ACESTEP_PROJECT_ROOT")
        os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
        return original

    def _restore_project_root(self, original: str | None) -> None:
        """Restore the project root environment variable."""
        if original is None:
            os.environ.pop("ACESTEP_PROJECT_ROOT", None)
        else:
            os.environ["ACESTEP_PROJECT_ROOT"] = original

    def test_startup_uses_gpu_optimization_defaults_without_user_preset(self) -> None:
        """Startup should not load a bundled preset over GPU-tier defaults."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            try:
                updates = premium_features.startup_preset_updates()
                names = premium_features.list_preset_names()
            finally:
                self._restore_project_root(original)

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(names, [])
        self.assertEqual(dropdown_update.get("value"), None)
        self.assertIn("GPU Optimization Preset", updates[len(keys) + 3])

    def test_no_system_preset_files_are_created(self) -> None:
        """The removed Premium Default/Turbo/Base files should not be regenerated."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            try:
                preset_dir = premium_features.ensure_default_preset()
                premium_features.startup_preset_updates()
            finally:
                self._restore_project_root(original)

        self.assertEqual(preset_dir.name, premium_features.USER_PRESET_FOLDER)
        self.assertFalse((Path(tmp_dir) / "premium_system_presets").exists())

    def test_user_preset_saves_and_loads_simple_model_selector(self) -> None:
        """User presets should still override GPU defaults when loaded."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("config_path")] = DEFAULT_TURBO_DIT_MODEL
            values[keys.index("simple_model_dropdown")] = DEFAULT_TURBO_DIT_MODEL
            values[keys.index("quantization_checkbox")] = "fp8_scaled"
            values[keys.index("simple_quantization")] = "fp8_scaled"
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("turbo fp8", None, *values)
                loaded = premium_features.load_named_preset("turbo fp8")
                updates = premium_features.load_preset_action("turbo fp8")
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        self.assertEqual(remembered, "turbo fp8")
        self.assertEqual(loaded["config_path"], DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(loaded["simple_model_dropdown"], DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(loaded["quantization_checkbox"], "fp8_scaled")
        self.assertEqual(loaded["simple_quantization"], "fp8_scaled")
        self.assertEqual(
            updates[keys.index("simple_model_dropdown")].get("value"),
            DEFAULT_TURBO_DIT_MODEL,
        )

    def test_user_preset_saves_and_loads_sam_trim_controls(self) -> None:
        """SAM-Audio trim checkbox and threshold should round-trip through presets."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("sam_trim_empty_output")] = True
            values[keys.index("sam_trim_threshold_db")] = -42.0
            try:
                premium_features.save_preset_action("sam trim", None, *values)
                loaded = premium_features.load_named_preset("sam trim")
                updates = premium_features.load_preset_action("sam trim")
            finally:
                self._restore_project_root(original)

        self.assertTrue(loaded["sam_trim_empty_output"])
        self.assertEqual(-42.0, loaded["sam_trim_threshold_db"])
        self.assertTrue(updates[keys.index("sam_trim_empty_output")].get("value"))
        self.assertEqual(
            -42.0,
            updates[keys.index("sam_trim_threshold_db")].get("value"),
        )

    def test_user_preset_saves_and_loads_extract_trim_controls(self) -> None:
        """ACE-Step Extract trim controls should round-trip through presets."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("extract_trim_empty_output")] = True
            values[keys.index("extract_trim_threshold_db")] = -44.0
            try:
                premium_features.save_preset_action("extract trim", None, *values)
                loaded = premium_features.load_named_preset("extract trim")
                updates = premium_features.load_preset_action("extract trim")
            finally:
                self._restore_project_root(original)

        self.assertTrue(loaded["extract_trim_empty_output"])
        self.assertEqual(-44.0, loaded["extract_trim_threshold_db"])
        self.assertTrue(updates[keys.index("extract_trim_empty_output")].get("value"))
        self.assertEqual(
            -44.0,
            updates[keys.index("extract_trim_threshold_db")].get("value"),
        )

    def test_user_preset_saves_and_loads_audio_processing_trim_controls(self) -> None:
        """Audio Processing trim controls should round-trip through presets."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("ap_trim_empty_output")] = True
            values[keys.index("ap_trim_threshold_db")] = -46.0
            values[keys.index("ap_trim_margin_seconds")] = 0.3
            values[keys.index("ap_trim_mincut")] = 20
            values[keys.index("ap_trim_minclip")] = 4
            try:
                premium_features.save_preset_action("ap trim", None, *values)
                loaded = premium_features.load_named_preset("ap trim")
                updates = premium_features.load_preset_action("ap trim")
            finally:
                self._restore_project_root(original)

        self.assertTrue(loaded["ap_trim_empty_output"])
        self.assertEqual(-46.0, loaded["ap_trim_threshold_db"])
        self.assertEqual(0.3, loaded["ap_trim_margin_seconds"])
        self.assertEqual(20, loaded["ap_trim_mincut"])
        self.assertEqual(4, loaded["ap_trim_minclip"])
        self.assertTrue(updates[keys.index("ap_trim_empty_output")].get("value"))
        self.assertEqual(
            -46.0,
            updates[keys.index("ap_trim_threshold_db")].get("value"),
        )
        self.assertEqual(
            0.3,
            updates[keys.index("ap_trim_margin_seconds")].get("value"),
        )
        self.assertEqual(20, updates[keys.index("ap_trim_mincut")].get("value"))
        self.assertEqual(4, updates[keys.index("ap_trim_minclip")].get("value"))

    def test_user_preset_clamps_trim_threshold_controls_to_supported_range(self) -> None:
        """Saved trim thresholds should load within the supported dB range."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("sam_trim_threshold_db")] = -120.0
            values[keys.index("extract_trim_threshold_db")] = 120.0
            values[keys.index("ap_trim_threshold_db")] = -120.0
            values[keys.index("ap_trim_margin_seconds")] = 20.0
            values[keys.index("ap_trim_mincut")] = 999
            values[keys.index("ap_trim_minclip")] = -5
            try:
                premium_features.save_preset_action("trim clamp", None, *values)
                loaded = premium_features.load_named_preset("trim clamp")
                updates = premium_features.load_preset_action("trim clamp")
            finally:
                self._restore_project_root(original)

        self.assertEqual(-100.0, loaded["sam_trim_threshold_db"])
        self.assertEqual(0.0, loaded["extract_trim_threshold_db"])
        self.assertEqual(-100.0, loaded["ap_trim_threshold_db"])
        self.assertEqual(5.0, loaded["ap_trim_margin_seconds"])
        self.assertEqual(300, loaded["ap_trim_mincut"])
        self.assertEqual(0, loaded["ap_trim_minclip"])
        self.assertEqual(
            -100.0,
            updates[keys.index("sam_trim_threshold_db")].get("value"),
        )
        self.assertEqual(
            0.0,
            updates[keys.index("extract_trim_threshold_db")].get("value"),
        )
        self.assertEqual(
            -100.0,
            updates[keys.index("ap_trim_threshold_db")].get("value"),
        )
        self.assertEqual(
            5.0,
            updates[keys.index("ap_trim_margin_seconds")].get("value"),
        )
        self.assertEqual(300, updates[keys.index("ap_trim_mincut")].get("value"))
        self.assertEqual(0, updates[keys.index("ap_trim_minclip")].get("value"))

    def test_startup_uses_remembered_existing_user_preset(self) -> None:
        """Startup should auto-load the last successfully saved user preset."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("config_path")] = DEFAULT_TURBO_DIT_MODEL
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("daily", None, *values)
                updates = premium_features.startup_preset_updates()
            finally:
                self._restore_project_root(original)

        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(updates[keys.index("config_path")].get("value"), DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(dropdown_update.get("value"), "daily")
        self.assertIn("Loaded preset: daily", updates[len(keys) + 3])

    def test_missing_remembered_preset_returns_to_gpu_defaults(self) -> None:
        """Missing last-used presets should clear selection and keep GPU defaults."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            try:
                premium_features.set_last_used_preset_name("deleted preset")
                updates = premium_features.startup_preset_updates()
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), None)
        self.assertIsNone(remembered)
        self.assertIn("missing", updates[len(keys) + 3])

    def test_corrupt_remembered_preset_returns_to_gpu_defaults(self) -> None:
        """Unreadable remembered presets should clear selection and keep GPU defaults."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            preset_dir = Path(tmp_dir) / premium_features.USER_PRESET_FOLDER
            preset_dir.mkdir(parents=True, exist_ok=True)
            (preset_dir / "broken.json").write_text("{not json", encoding="utf-8")
            try:
                premium_features.set_last_used_preset_name("broken")
                updates = premium_features.startup_preset_updates()
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), None)
        self.assertIsNone(remembered)
        self.assertIn("Using GPU Optimization Preset", updates[len(keys) + 3])

    def test_load_missing_preset_returns_to_gpu_defaults(self) -> None:
        """Manual loads of missing presets should not apply a bundled fallback."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            try:
                updates = premium_features.load_preset_action("does not exist")
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), None)
        self.assertIsNone(remembered)
        self.assertIn("missing", updates[len(keys) + 3])

    def test_delete_preset_selects_next_available_preset(self) -> None:
        """Deleting the current preset should load the next preset in dropdown order."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            default_values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            alpha_values = list(default_values)
            beta_values = list(default_values)
            gamma_values = list(default_values)
            alpha_values[keys.index("captions")] = "alpha caption"
            beta_values[keys.index("captions")] = "beta caption"
            gamma_values[keys.index("captions")] = "gamma caption"
            try:
                premium_features.save_preset_action("alpha", None, *alpha_values)
                premium_features.save_preset_action("beta", None, *beta_values)
                premium_features.save_preset_action("gamma", None, *gamma_values)
                updates = premium_features.delete_preset_action(
                    "beta",
                    default_values,
                )
                names = premium_features.list_preset_names()
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(names, ["alpha", "gamma"])
        self.assertEqual(dropdown_update.get("choices"), ["alpha", "gamma"])
        self.assertEqual(dropdown_update.get("value"), "gamma")
        self.assertEqual(remembered, "gamma")
        self.assertEqual(updates[keys.index("captions")].get("value"), "gamma caption")
        self.assertIn("Loaded next preset: gamma", updates[len(keys) + 3])

    def test_delete_preset_falls_back_to_typed_name(self) -> None:
        """Delete should still work if the browser sends no dropdown value."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            default_values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            try:
                premium_features.save_preset_action("typed only", None, *default_values)
                updates = premium_features.delete_preset_action(
                    None,
                    default_values,
                    "typed only",
                )
                names = premium_features.list_preset_names()
            finally:
                self._restore_project_root(original)

        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(names, [])
        self.assertEqual(dropdown_update.get("choices"), [])
        self.assertIsNone(dropdown_update.get("value"))
        self.assertIn("Deleted preset: typed only", updates[len(keys) + 3])

    def test_delete_last_preset_restores_gradio_defaults(self) -> None:
        """Deleting the final preset should clear selection and reset all fields."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            default_values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            default_values[keys.index("captions")] = "default caption"
            default_values[keys.index("training_tensor_dir")] = "./datasets/preprocessed_tensors"
            preset_values = list(default_values)
            preset_values[keys.index("captions")] = "custom caption"
            preset_values[keys.index("training_tensor_dir")] = r"G:\custom\tensors"
            try:
                premium_features.save_preset_action("only", None, *preset_values)
                updates = premium_features.delete_preset_action(
                    "only",
                    default_values,
                )
                names = premium_features.list_preset_names()
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(names, [])
        self.assertEqual(dropdown_update.get("choices"), [])
        self.assertIsNone(dropdown_update.get("value"))
        self.assertIsNone(remembered)
        self.assertEqual(updates[keys.index("captions")].get("value"), "default caption")
        self.assertEqual(
            updates[keys.index("training_tensor_dir")].get("value"),
            "./datasets/preprocessed_tensors",
        )
        self.assertIn("Using GPU Optimization Preset", updates[len(keys) + 3])

    def test_previous_system_names_are_allowed_as_user_presets(self) -> None:
        """Removed system preset names should behave like ordinary user names."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("Premium Default", None, *values)
                names = premium_features.list_preset_names()
            finally:
                self._restore_project_root(original)

        self.assertIn("Premium Default", names)

    def test_user_preset_without_checkpoint_backfills_install_local_models_dir(self) -> None:
        """Older presets missing checkpoint info should backfill runtime defaults."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            preset_dir = Path(tmp_dir) / premium_features.USER_PRESET_FOLDER
            preset_dir.mkdir(parents=True, exist_ok=True)
            (preset_dir / "legacy.json").write_text(
                json.dumps({"values": {"config_path": "acestep-v15-xl-sft"}}),
                encoding="utf-8",
            )
            expected = str(get_models_dir(project_root=tmp_dir))
            try:
                payload = premium_features.load_named_preset("legacy")
            finally:
                self._restore_project_root(original)

        self.assertEqual(payload["checkpoint_dropdown"], expected)
        self.assertEqual(payload["config_path"], DEFAULT_PREMIUM_DIT_MODEL)
        self.assertEqual(payload["simple_model_dropdown"], DEFAULT_PREMIUM_DIT_MODEL)
        self.assertEqual(payload["inference_steps"], 50)
        self.assertEqual(payload["guidance_scale"], 7.0)
        self.assertFalse(payload["use_adg"])
        self.assertEqual(payload["shift"], 3.0)
        self.assertTrue(payload["init_lm_checkbox"])
        self.assertTrue(payload["think_checkbox"])
        self.assertTrue(payload["use_cot_metas"])
        self.assertFalse(payload["dcw_enabled"])
        self.assertEqual(payload["dcw_scaler"], 0.0)
        self.assertEqual(payload["dcw_high_scaler"], 0.0)

    def test_user_preset_without_quality_values_uses_model_defaults(self) -> None:
        """Legacy SFT/Base presets should not inherit Turbo's 8-step values."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            preset_dir = Path(tmp_dir) / premium_features.USER_PRESET_FOLDER
            preset_dir.mkdir(parents=True, exist_ok=True)
            (preset_dir / "base.json").write_text(
                json.dumps({"values": {"config_path": DEFAULT_BASE_DIT_MODEL}}),
                encoding="utf-8",
            )
            (preset_dir / "sft.json").write_text(
                json.dumps({"values": {"config_path": DEFAULT_PREMIUM_DIT_MODEL}}),
                encoding="utf-8",
            )
            (preset_dir / "legacy_sft.json").write_text(
                json.dumps({"values": {"config_path": "acestep-v15-sft"}}),
                encoding="utf-8",
            )
            try:
                base_payload = premium_features.load_named_preset("base")
                sft_payload = premium_features.load_named_preset("sft")
                legacy_sft_payload = premium_features.load_named_preset("legacy_sft")
                base_updates = premium_features.load_preset_action("base")
                sft_updates = premium_features.load_preset_action("sft")
            finally:
                self._restore_project_root(original)

        keys = premium_features.get_preset_component_keys()
        for payload in (sft_payload, legacy_sft_payload):
            self.assertEqual(payload["inference_steps"], 50)
            self.assertEqual(payload["guidance_scale"], 7.0)
            self.assertFalse(payload["use_adg"])
            self.assertEqual(payload["shift"], 3.0)
            self.assertTrue(payload["init_lm_checkbox"])
            self.assertTrue(payload["think_checkbox"])
            self.assertTrue(payload["use_cot_metas"])
            self.assertFalse(payload["allow_lm_batch"])
            self.assertFalse(payload["dcw_enabled"])
            self.assertEqual(payload["dcw_scaler"], 0.0)
            self.assertEqual(payload["dcw_high_scaler"], 0.0)

        self.assertEqual(base_payload["inference_steps"], 64)
        self.assertEqual(base_payload["guidance_scale"], 7.0)
        self.assertFalse(base_payload["use_adg"])
        self.assertEqual(base_payload["shift"], 3.0)
        self.assertFalse(base_payload["dcw_enabled"])
        self.assertEqual(base_payload["dcw_scaler"], 0.0)
        self.assertEqual(base_payload["dcw_high_scaler"], 0.0)

        self.assertEqual(base_updates[keys.index("inference_steps")].get("value"), 64)
        self.assertEqual(base_updates[keys.index("inference_steps")].get("maximum"), 200)
        self.assertTrue(base_updates[keys.index("guidance_scale")].get("visible"))
        self.assertFalse(base_updates[keys.index("use_adg")].get("value"))
        self.assertFalse(base_updates[keys.index("dcw_enabled")].get("value"))
        self.assertEqual(sft_updates[keys.index("shift")].get("value"), 3.0)
        self.assertEqual(sft_updates[keys.index("shift")].get("minimum"), 1.0)
        self.assertEqual(sft_updates[keys.index("shift")].get("maximum"), 5.0)
        self.assertEqual(sft_updates[keys.index("inference_steps")].get("value"), 50)
        self.assertFalse(sft_updates[keys.index("use_adg")].get("value"))
        self.assertFalse(sft_updates[keys.index("dcw_enabled")].get("value"))
        self.assertEqual(sft_updates[keys.index("dcw_scaler")].get("value"), 0.0)
        self.assertEqual(sft_updates[keys.index("dcw_high_scaler")].get("value"), 0.0)
        self.assertTrue(sft_updates[keys.index("guidance_scale")].get("visible"))

    def test_user_preset_keeps_custom_quality_and_vram_settings_selected(self) -> None:
        """User presets should display as selected and override auto VRAM defaults."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("config_path")] = DEFAULT_PREMIUM_DIT_MODEL
            values[keys.index("simple_model_dropdown")] = DEFAULT_PREMIUM_DIT_MODEL
            values[keys.index("tier_dropdown")] = "tier6b"
            values[keys.index("offload_to_cpu_checkbox")] = True
            values[keys.index("offload_dit_to_cpu_checkbox")] = False
            values[keys.index("quantization_checkbox")] = "fp8_scaled"
            values[keys.index("simple_quantization")] = "fp8_scaled"
            values[keys.index("lm_model_path")] = "acestep-5Hz-lm-1.7B"
            values[keys.index("backend_dropdown")] = "pt"
            values[keys.index("init_llm_checkbox")] = True
            values[keys.index("batch_size_input")] = 1
            values[keys.index("inference_steps")] = 88
            values[keys.index("guidance_scale")] = 8.5
            values[keys.index("use_adg")] = False
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("custom high", None, *values)
                updates = premium_features.load_preset_action("custom high")
            finally:
                self._restore_project_root(original)

        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), "custom high")
        self.assertEqual(updates[keys.index("tier_dropdown")].get("value"), "tier6b")
        self.assertEqual(updates[keys.index("offload_to_cpu_checkbox")].get("value"), True)
        self.assertEqual(updates[keys.index("quantization_checkbox")].get("value"), "fp8_scaled")
        self.assertEqual(updates[keys.index("simple_quantization")].get("value"), "fp8_scaled")
        self.assertEqual(updates[keys.index("lm_model_path")].get("value"), "acestep-5Hz-lm-1.7B")
        self.assertEqual(updates[keys.index("inference_steps")].get("value"), 88)
        self.assertEqual(updates[keys.index("guidance_scale")].get("value"), 8.5)
        self.assertEqual(updates[keys.index("use_adg")].get("value"), False)

    def test_user_preset_saves_and_loads_lora_selection(self) -> None:
        """LoRA path/dropdown/scale should persist through user preset save/load."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            adapter = _write_peft_adapter(Path(tmp_dir) / "Loras" / "voice")
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("lora_dropdown")] = str(adapter)
            values[keys.index("lora_scale_slider")] = 0.65
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("lora preset", None, *values)
                loaded = premium_features.load_named_preset("lora preset")
                updates = premium_features.load_preset_action("lora preset")
            finally:
                self._restore_project_root(original)

        self.assertEqual(loaded["lora_dropdown"], str(adapter))
        self.assertEqual(loaded["simple_lora_dropdown"], str(adapter))
        self.assertEqual(loaded["lora_scale_slider"], 0.65)
        self.assertEqual(loaded["simple_lora_scale_slider"], 0.65)
        self.assertEqual(updates[keys.index("lora_dropdown")].get("value"), str(adapter))
        self.assertEqual(
            updates[keys.index("simple_lora_dropdown")].get("value"),
            str(adapter),
        )
        self.assertIn("Next run will use LoRA:", updates[len(keys)])

    def test_user_preset_saves_and_loads_lora_training_parameters(self) -> None:
        """LoRA training tab controls should round-trip through user presets."""

        expected = {
            "training_tensor_dir": r"G:\data\tensors",
            "lora_model_config": DEFAULT_PREMIUM_DIT_MODEL,
            "lora_vram_preset": "Manual",
            "lora_name": "voice-style",
            "lora_rank": 36,
            "lora_alpha": 72,
            "lora_dropout": 0.15,
            "learning_rate": 0.00022,
            "train_epochs": 321,
            "train_batch_size": 3,
            "gradient_accumulation": 5,
            "save_every_n_epochs": 7,
            "lora_save_best_after": 12,
            "lora_save_best_smoothing_window": 3,
            "lora_save_best_min_delta": 0.002,
            "training_shift": 4.25,
            "training_num_inference_steps": 37,
            "training_seed": 12345,
            "lora_optimizer_type": "adamw8bit",
            "lora_scheduler_type": "constant",
            "lora_validation_split_percent": 15,
            "lora_output_dir": r"G:\loras",
            "resume_checkpoint_dir": r"G:\loras\voice-style\epoch-7-training_resume_state.pt",
            "lora_gradient_checkpointing": False,
            "lora_activation_cpu_offload": True,
            "lora_offload_non_decoder": False,
            "lora_keep_frozen_bf16": True,
            "lora_base_quantization": "FP8 scaled",
            "lora_empty_cache_every_n_steps": 19,
            "lora_sample_enabled": True,
            "lora_sample_every_n_epochs": 4,
            "lora_sample_prompt": "training sample prompt",
            "lora_sample_lyrics": "[verse]\ntraining sample lyrics",
            "lora_sample_seed": 9876,
            "lora_sample_offload_training_model": True,
            "lora_sample_offload_generation": False,
            "training_subprocess": False,
        }
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            for key, value in expected.items():
                values[keys.index(key)] = value
            try:
                premium_features.save_preset_action("lora training", None, *values)
                loaded = premium_features.load_named_preset("lora training")
                updates = premium_features.load_preset_action("lora training")
            finally:
                self._restore_project_root(original)

        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(loaded[key], value)
                self.assertEqual(updates[keys.index(key)].get("value"), value)

    def test_legacy_lora_8bit_checkbox_migrates_to_optimizer_dropdown(self) -> None:
        """Older LoRA presets should map the removed checkbox into optimizer type."""

        migrated = premium_features._apply_runtime_defaults(
            {"lora_use_8bit_adam": True}
        )
        explicit = premium_features._apply_runtime_defaults(
            {
                "lora_use_8bit_adam": True,
                "lora_optimizer_type": "adafactor",
            }
        )

        self.assertEqual("adamw8bit", migrated["lora_optimizer_type"])
        self.assertEqual("adafactor", explicit["lora_optimizer_type"])

    def test_user_preset_preserves_saved_blank_text_values(self) -> None:
        """Saved empty text fields should not be replaced by built-in defaults."""

        blank_keys = (
            "captions",
            "lyrics",
            "simple_create_caption",
            "simple_create_lyrics",
            "lora_sample_prompt",
            "lora_sample_lyrics",
            "flow_edit_source_caption",
            "flow_edit_source_lyrics",
        )
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            for key in blank_keys:
                values[keys.index(key)] = ""
            try:
                premium_features.save_preset_action("blank text", None, *values)
                loaded = premium_features.load_named_preset("blank text")
                updates = premium_features.load_preset_action("blank text")
            finally:
                self._restore_project_root(original)

        for key in blank_keys:
            with self.subTest(key=key):
                self.assertEqual(loaded[key], "")
                self.assertEqual(updates[keys.index(key)].get("value"), "")

    def test_user_preset_file_upload_fields_are_gradio_safe(self) -> None:
        """Empty or directory upload values should load as empty file components."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            reference_file = Path(tmp_dir) / "reference.wav"
            reference_file.write_bytes(b"RIFF")
            values[keys.index("reference_audio")] = str(reference_file)
            values[keys.index("src_audio")] = tmp_dir
            values[keys.index("lm_codes_audio_upload")] = ""
            values[keys.index("simple_create_cover_image")] = ""
            try:
                premium_features.save_preset_action("file fields", None, *values)
                updates = premium_features.load_preset_action("file fields")
            finally:
                self._restore_project_root(original)

        self.assertEqual(
            updates[keys.index("reference_audio")].get("value"),
            str(reference_file),
        )
        self.assertIsNone(updates[keys.index("src_audio")].get("value"))
        self.assertIsNone(updates[keys.index("lm_codes_audio_upload")].get("value"))
        self.assertIsNone(updates[keys.index("simple_create_cover_image")].get("value"))

    def test_user_preset_values_are_sanitized_for_gradio_components(self) -> None:
        """Malformed saved values should not be pushed back into strict controls."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("train_epochs")] = 100200
            values[keys.index("lokr_export_epoch")] = "Latest (Auto)"
            component_specs = {
                "train_epochs": {
                    "component_type": "Slider",
                    "minimum": 1,
                    "maximum": 4000,
                    "value": 100,
                },
                "lokr_export_epoch": {
                    "component_type": "Dropdown",
                    "choices": ["Latest (auto)"],
                    "value": "Latest (auto)",
                },
            }
            try:
                premium_features.save_preset_action("malformed", None, *values)
                loaded = premium_features.load_named_preset("malformed")
                updates = premium_features.load_preset_action(
                    "malformed",
                    component_specs,
                )
            finally:
                self._restore_project_root(original)

        self.assertEqual(loaded["train_epochs"], 100200)
        self.assertEqual(loaded["lokr_export_epoch"], "Latest (Auto)")
        self.assertEqual(updates[keys.index("train_epochs")].get("value"), 4000)
        self.assertEqual(
            updates[keys.index("lokr_export_epoch")].get("value"),
            "Latest (auto)",
        )


if __name__ == "__main__":
    unittest.main()
