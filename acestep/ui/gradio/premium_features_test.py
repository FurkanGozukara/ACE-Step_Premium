"""Tests for user presets and GPU-optimization startup defaults."""

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
    get_models_dir,
)
from acestep.ui.gradio import premium_features
from acestep.ui.gradio.events.generation.remix_presets import (
    REMIX_PRESET_TRANSLATION,
)


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
        self.assertEqual(
            premium_features.DEFAULT_PRESET_VALUES["config_path"],
            DEFAULT_TURBO_DIT_MODEL,
        )
        self.assertEqual(
            premium_features.DEFAULT_PRESET_VALUES["simple_model_dropdown"],
            DEFAULT_TURBO_DIT_MODEL,
        )
        self.assertEqual(premium_features.DEFAULT_PRESET_VALUES["vocal_language"], "en")
        self.assertEqual(
            premium_features.DEFAULT_PRESET_VALUES["simple_vocal_language"],
            "en",
        )
        self.assertEqual(
            premium_features.DEFAULT_PRESET_VALUES["generation_mode"],
            "Remix",
        )

    def test_runtime_defaults_align_mode_task_type_and_retention(self) -> None:
        """Loaded preset defaults should not leave Remix retention on text generation."""

        custom = premium_features._apply_runtime_defaults(
            {
                "generation_mode": "Custom",
                "task_type": "cover",
                "cover_noise_strength": 0.97,
            }
        )
        remix = premium_features._apply_runtime_defaults(
            {
                "generation_mode": "Remix",
                "task_type": "text2music",
            }
        )

        self.assertEqual(custom["task_type"], "text2music")
        self.assertEqual(custom["cover_noise_strength"], 0.0)
        self.assertEqual(remix["task_type"], "cover")
        self.assertGreater(remix["cover_noise_strength"], 0.0)

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

    def test_user_preset_save_syncs_vocal_language_keys(self) -> None:
        """Saved preset JSON should keep simple and advanced language keys aligned."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("vocal_language")] = "ja"
            values[keys.index("simple_vocal_language")] = "unknown"
            values[keys.index("simple_create_vocal_language")] = "unknown"
            try:
                premium_features.save_preset_action("language sync", None, *values)
                preset_path = premium_features._user_preset_path("language sync")
                stored = json.loads(preset_path.read_text(encoding="utf-8"))["values"]
                loaded = premium_features.load_named_preset("language sync")
            finally:
                self._restore_project_root(original)

        for payload in (stored, loaded):
            self.assertEqual(payload["vocal_language"], "ja")
            self.assertEqual(payload["simple_vocal_language"], "ja")
            self.assertEqual(payload["simple_create_vocal_language"], "ja")

    def test_user_preset_saves_and_loads_remix_preset_selector(self) -> None:
        """Remix preset selection and paired strength values should round-trip."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("generation_mode")] = "Remix"
            values[keys.index("remix_preset")] = REMIX_PRESET_TRANSLATION
            values[keys.index("audio_cover_strength")] = 0.70
            values[keys.index("cover_noise_strength")] = 0.20
            try:
                premium_features.save_preset_action("translation remix", None, *values)
                loaded = premium_features.load_named_preset("translation remix")
                updates = premium_features.load_preset_action("translation remix")
            finally:
                self._restore_project_root(original)

        self.assertEqual(loaded["remix_preset"], REMIX_PRESET_TRANSLATION)
        self.assertEqual(loaded["audio_cover_strength"], 0.70)
        self.assertEqual(loaded["cover_noise_strength"], 0.20)
        self.assertEqual(
            updates[keys.index("remix_preset")].get("value"),
            REMIX_PRESET_TRANSLATION,
        )
        self.assertTrue(updates[keys.index("remix_preset")].get("visible"))
        self.assertNotIn(
            "ace-mode-hidden",
            updates[keys.index("remix_preset")].get("elem_classes"),
        )
        self.assertEqual(
            updates[keys.index("audio_cover_strength")].get("value"),
            0.70,
        )
        self.assertEqual(
            updates[keys.index("cover_noise_strength")].get("value"),
            0.20,
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

    def test_user_preset_saves_and_loads_extract_all_stems(self) -> None:
        """Extract-all-stems should round-trip through custom user presets."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("extract_all_stems")] = True
            try:
                premium_features.save_preset_action("extract all stems", None, *values)
                loaded = premium_features.load_named_preset("extract all stems")
                updates = premium_features.load_preset_action("extract all stems")
            finally:
                self._restore_project_root(original)

        self.assertTrue(loaded["extract_all_stems"])
        self.assertTrue(updates[keys.index("extract_all_stems")].get("value"))

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
            values[keys.index("ap_export_audio_only")] = True
            values[keys.index("ap_run_subprocess")] = False
            values[keys.index("ap_disable_upload_preview")] = True
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
        self.assertTrue(loaded["ap_export_audio_only"])
        self.assertFalse(loaded["ap_run_subprocess"])
        self.assertTrue(loaded["ap_disable_upload_preview"])
        self.assertEqual(-46.0, loaded["ap_trim_threshold_db"])
        self.assertEqual(0.3, loaded["ap_trim_margin_seconds"])
        self.assertEqual(20, loaded["ap_trim_mincut"])
        self.assertEqual(4, loaded["ap_trim_minclip"])
        self.assertTrue(updates[keys.index("ap_trim_empty_output")].get("value"))
        self.assertTrue(updates[keys.index("ap_export_audio_only")].get("value"))
        self.assertFalse(updates[keys.index("ap_run_subprocess")].get("value"))
        self.assertTrue(updates[keys.index("ap_disable_upload_preview")].get("value"))
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

    def test_startup_uses_defaults_instead_of_remembered_user_preset(self) -> None:
        """Startup should not auto-load the last successfully saved user preset."""
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("config_path")] = DEFAULT_BASE_DIT_MODEL
            values[keys.index("simple_model_dropdown")] = DEFAULT_BASE_DIT_MODEL
            values[keys.index("generation_mode")] = "Complete"
            values[keys.index("vocal_language")] = "ja"
            values[keys.index("simple_vocal_language")] = "ja"
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("daily", None, *values)
                updates = premium_features.startup_preset_updates()
                remembered = premium_features.get_last_used_preset_name()
            finally:
                self._restore_project_root(original)

        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(updates[keys.index("config_path")], premium_features.gr.skip())
        self.assertEqual(updates[keys.index("generation_mode")], premium_features.gr.skip())
        self.assertEqual(updates[keys.index("vocal_language")], premium_features.gr.skip())
        self.assertEqual(dropdown_update.get("choices"), ["daily"])
        self.assertEqual(dropdown_update.get("value"), None)
        self.assertIsNone(remembered)
        self.assertIn("Using GPU Optimization Preset", updates[len(keys) + 3])

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
        self.assertIn("Using GPU Optimization Preset", updates[len(keys) + 3])

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
        self.assertFalse(payload["use_cot_caption"])
        self.assertFalse(payload["use_cot_language"])
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
            self.assertFalse(payload["use_cot_caption"])
            self.assertFalse(payload["use_cot_language"])
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

    def test_user_preset_syncs_generate_song_and_advanced_vocal_language(self) -> None:
        """Saved presets should load one shared vocal-language value across tabs."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("vocal_language")] = "tr"
            values[keys.index("simple_vocal_language")] = "tr"
            values[keys.index("simple_create_vocal_language")] = "tr"
            try:
                premium_features.save_preset_action("language sync", None, *values)
                loaded = premium_features.load_named_preset("language sync")
                updates = premium_features.load_preset_action("language sync")
            finally:
                self._restore_project_root(original)

        self.assertEqual(loaded["vocal_language"], "tr")
        self.assertEqual(loaded["simple_vocal_language"], "tr")
        self.assertEqual(loaded["simple_create_vocal_language"], "tr")
        self.assertEqual(updates[keys.index("vocal_language")].get("value"), "tr")
        self.assertEqual(updates[keys.index("simple_vocal_language")].get("value"), "tr")
        self.assertEqual(
            updates[keys.index("simple_create_vocal_language")].get("value"),
            "tr",
        )

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
        self.assertEqual(
            updates[keys.index("simple_create_tier_dropdown")].get("value"),
            "tier6b",
        )
        self.assertEqual(updates[keys.index("offload_to_cpu_checkbox")].get("value"), True)
        self.assertEqual(updates[keys.index("quantization_checkbox")].get("value"), "fp8_scaled")
        self.assertEqual(updates[keys.index("simple_quantization")].get("value"), "fp8_scaled")
        self.assertEqual(updates[keys.index("lm_model_path")].get("value"), "acestep-5Hz-lm-1.7B")
        self.assertEqual(updates[keys.index("inference_steps")].get("value"), 88)
        self.assertEqual(updates[keys.index("guidance_scale")].get("value"), 8.5)
        self.assertEqual(updates[keys.index("use_adg")].get("value"), False)

    def test_legacy_removed_vram_tier_preset_uses_detected_gpu_tier(self) -> None:
        """Removed tier2/tier3 preset values should migrate to the auto GPU tier."""

        fake_gpu_config = SimpleNamespace(
            tier="unlimited",
            mlx_vae_chunk_size=1024,
            recommended_lm_model="acestep-5Hz-lm-4B",
        )
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("tier_dropdown")] = "tier2"
            values[keys.index("simple_create_tier_dropdown")] = "tier3"
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                with patch.object(
                    premium_features,
                    "get_global_gpu_config",
                    return_value=fake_gpu_config,
                ):
                    premium_features.save_preset_action("legacy vram", None, *values)
                    loaded = premium_features.load_named_preset("legacy vram")
                    updates = premium_features.load_preset_action("legacy vram")
            finally:
                self._restore_project_root(original)

        self.assertEqual(loaded["tier_dropdown"], "unlimited")
        self.assertEqual(loaded["simple_create_tier_dropdown"], "unlimited")
        self.assertEqual(updates[keys.index("tier_dropdown")].get("value"), "unlimited")
        self.assertEqual(
            updates[keys.index("simple_create_tier_dropdown")].get("value"),
            "unlimited",
        )

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
            "lora_weight_decay": 0.02,
            "lora_adam_beta1": 0.85,
            "lora_adam_beta2": 0.995,
            "lora_adam_epsilon": 1e-7,
            "lora_adamw8bit_min_8bit_size": 2048,
            "lora_adamw8bit_percentile_clipping": 95,
            "lora_adamw8bit_block_wise": False,
            "lora_adamw8bit_paged": True,
            "lora_adafactor_epsilon1": 1e-29,
            "lora_adafactor_epsilon2": 1e-2,
            "lora_adafactor_clip_threshold": 1.5,
            "lora_adafactor_decay_rate": -0.7,
            "lora_adafactor_beta1": 0.1,
            "lora_adafactor_scale_parameter": True,
            "lora_adafactor_relative_step": True,
            "lora_adafactor_warmup_init": True,
            "lora_scheduler_type": "constant",
            "lora_validation_split_percent": 15,
            "lora_output_dir": r"G:\loras",
            "resume_checkpoint_dir": r"G:\loras\voice-style\epoch-7-training_resume_state.pt",
            "lora_gradient_checkpointing": False,
            "lora_activation_cpu_offload": True,
            "lora_offload_non_decoder": False,
            "lora_keep_frozen_bf16": True,
            "lora_compile_model": True,
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
        self.assertTrue(
            updates[keys.index("lora_adamw8bit_min_8bit_size")].get("visible")
        )
        self.assertFalse(updates[keys.index("lora_adafactor_epsilon1")].get("visible"))
        self.assertEqual(
            expected["lora_adafactor_epsilon1"],
            updates[keys.index("lora_adafactor_epsilon1")].get("value"),
        )

    def test_user_preset_file_contains_every_tracked_setting_key(self) -> None:
        """Saving a preset should serialize the complete custom-config schema."""

        overrides = {
            "sam_batch_overwrite_existing": True,
            "batch_auto_improve_lyrics": True,
            "batch_auto_improve_style": True,
            "lora_scheduler_type": "linear",
            "lora_optimizer_type": "adafactor",
            "lora_adafactor_relative_step": False,
            "sam_output_format": "flac",
            "extract_output_format": "wav",
            "audio_format": "wav",
            "training_subprocess": False,
        }
        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            for key, value in overrides.items():
                values[keys.index(key)] = value
            try:
                premium_features.save_preset_action("all settings", None, *values)
                preset_path = (
                    Path(tmp_dir)
                    / premium_features.USER_PRESET_FOLDER
                    / "all settings.json"
                )
                saved = json.loads(preset_path.read_text(encoding="utf-8"))["values"]
                loaded = premium_features.load_named_preset("all settings")
                updates = premium_features.load_preset_action("all settings")
            finally:
                self._restore_project_root(original)

        self.assertEqual([], [key for key, count in Counter(keys).items() if count > 1])
        self.assertEqual(set(keys), set(saved))
        for key, value in overrides.items():
            with self.subTest(key=key):
                self.assertEqual(loaded[key], value)
                self.assertEqual(updates[keys.index(key)].get("value"), value)

    def test_default_lora_training_adapter_is_dora(self) -> None:
        """New training presets should select DoRA by default."""

        self.assertEqual(
            "dora",
            premium_features.DEFAULT_PRESET_VALUES["lora_adapter_type"],
        )

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
        self.assertEqual(0.0, explicit["lora_weight_decay"])
        self.assertEqual(1e-30, explicit["lora_adafactor_epsilon1"])

    def test_loading_adafactor_preset_updates_optimizer_field_values(self) -> None:
        """Custom presets should restore optimizer hyperparameter values."""

        with self._with_project_root() as tmp_dir:
            original = self._set_project_root(tmp_dir)
            keys = premium_features.get_preset_component_keys()
            values = [
                premium_features.DEFAULT_PRESET_VALUES.get(key, "")
                for key in keys
            ]
            values[keys.index("lora_optimizer_type")] = "adafactor"
            values[keys.index("lora_weight_decay")] = 0.0
            values[keys.index("lora_adafactor_beta1")] = 0.12
            values[keys.index("lora_adafactor_relative_step")] = False
            values[keys.index("lora_adafactor_scale_parameter")] = False
            values[keys.index("lora_adafactor_warmup_init")] = False
            try:
                premium_features.save_preset_action("adafactor training", None, *values)
                updates = premium_features.load_preset_action("adafactor training")
                optimizer_updates = (
                    premium_features.load_lora_optimizer_hyperparameter_updates_for_preset(
                        "adafactor training"
                    )
                )
            finally:
                self._restore_project_root(original)

        self.assertEqual(
            "adafactor",
            updates[keys.index("lora_optimizer_type")].get("value"),
        )
        self.assertTrue(updates[keys.index("lora_weight_decay")].get("visible"))
        self.assertTrue(updates[keys.index("lora_adafactor_epsilon1")].get("visible"))
        self.assertTrue(
            updates[keys.index("lora_adafactor_relative_step")].get("visible")
        )
        self.assertTrue(updates[keys.index("lora_adam_beta1")].get("visible"))
        self.assertTrue(
            updates[keys.index("lora_adamw8bit_min_8bit_size")].get("visible")
        )
        optimizer_update_map = dict(
            zip(premium_features._LORA_OPTIMIZER_PRESET_KEY_MAP, optimizer_updates)
        )
        self.assertEqual(
            0.12,
            optimizer_update_map["lora_adafactor_beta1"].get("value"),
        )
        self.assertTrue(
            optimizer_update_map["lora_adafactor_beta1"].get("visible")
        )

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
