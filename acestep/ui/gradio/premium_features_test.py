"""Tests for user presets and GPU-optimization startup defaults."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from acestep.model_downloader import DEFAULT_TURBO_DIT_MODEL, get_models_dir
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
        self.assertEqual(payload["config_path"], "acestep-v15-xl-sft")
        self.assertEqual(payload["simple_model_dropdown"], "acestep-v15-xl-sft")

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
        self.assertEqual(loaded["lora_scale_slider"], 0.65)
        self.assertEqual(updates[keys.index("lora_dropdown")].get("value"), str(adapter))
        self.assertIn("Next run will use LoRA:", updates[len(keys)])


if __name__ == "__main__":
    unittest.main()
