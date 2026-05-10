"""Tests for premium preset defaults and install-local path resolution."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
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
    """Verify preset loading keeps install-local defaults stable."""

    def test_default_preset_loads_install_local_checkpoint(self):
        """Default preset should resolve checkpoint path from current install root."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            expected = str(get_models_dir(project_root=tmp_dir))
            try:
                payload = premium_features.load_named_preset(
                    premium_features.DEFAULT_PRESET_NAME
                )
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(payload["checkpoint_dropdown"], expected)
        self.assertEqual(payload["captions"], premium_features.DEFAULT_PRESET_CAPTION)
        self.assertEqual(payload["lyrics"], premium_features.DEFAULT_PRESET_LYRICS)
        self.assertEqual(payload["audio_format"], "flac_mp3")
        self.assertEqual(payload["mp3_bitrate"], "320k")
        self.assertEqual(payload["simple_model_dropdown"], "acestep-v15-xl-sft")
        self.assertEqual(payload["simple_quantization"], "none")
        self.assertEqual(payload["inference_steps"], 50)
        self.assertEqual(payload["guidance_scale"], 7.0)
        self.assertEqual(payload["shift"], 3.0)
        self.assertEqual(payload["lora_path"], "")
        self.assertEqual(payload["lora_dropdown"], "")
        self.assertEqual(payload["lora_scale_slider"], 1.0)

    def test_default_preset_lives_in_system_folder(self):
        """Built-in defaults should be stored away from user preset files."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                target = premium_features.ensure_default_preset()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(target.parent.name, premium_features.DEFAULT_PRESET_FOLDER)
        self.assertNotEqual(
            premium_features.DEFAULT_PRESET_FOLDER,
            premium_features.USER_PRESET_FOLDER,
        )

    def test_system_presets_include_default_turbo(self):
        """The protected preset list should expose the XL-Turbo preset."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            expected = str(get_models_dir(project_root=tmp_dir))
            try:
                names = premium_features.list_preset_names()
                turbo_payload = premium_features.load_named_preset(
                    premium_features.DEFAULT_TURBO_PRESET_NAME
                )
                turbo_path = (
                    Path(tmp_dir)
                    / premium_features.DEFAULT_PRESET_FOLDER
                    / "default_turbo.json"
                )
                raw_payload = json.loads(turbo_path.read_text(encoding="utf-8"))
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(
            names[:2],
            [
                premium_features.DEFAULT_PRESET_NAME,
                premium_features.DEFAULT_TURBO_PRESET_NAME,
            ],
        )
        self.assertEqual(turbo_payload["checkpoint_dropdown"], expected)
        self.assertEqual(turbo_payload["config_path"], DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(
            turbo_payload["simple_model_dropdown"],
            DEFAULT_TURBO_DIT_MODEL,
        )
        self.assertEqual(turbo_payload["simple_quantization"], "none")
        self.assertEqual(turbo_payload["inference_steps"], 8)
        self.assertEqual(turbo_payload["guidance_scale"], 1.0)
        self.assertEqual(turbo_payload["shift"], 3.0)
        self.assertFalse(turbo_payload["use_adg"])
        self.assertTrue(raw_payload["_meta"]["immutable"])
        self.assertEqual(
            raw_payload["_meta"]["name"],
            premium_features.DEFAULT_TURBO_PRESET_NAME,
        )

    def test_system_presets_include_default_base(self):
        """The protected preset list should expose the XL-Base preset."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            expected = str(get_models_dir(project_root=tmp_dir))
            try:
                names = premium_features.list_preset_names()
                base_payload = premium_features.load_named_preset(
                    premium_features.DEFAULT_BASE_PRESET_NAME
                )
                base_path = (
                    Path(tmp_dir)
                    / premium_features.DEFAULT_PRESET_FOLDER
                    / "default_base.json"
                )
                raw_payload = json.loads(base_path.read_text(encoding="utf-8"))
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertIn(premium_features.DEFAULT_BASE_PRESET_NAME, names)
        self.assertEqual(base_payload["checkpoint_dropdown"], expected)
        self.assertEqual(base_payload["config_path"], DEFAULT_BASE_DIT_MODEL)
        self.assertEqual(
            base_payload["simple_model_dropdown"],
            DEFAULT_BASE_DIT_MODEL,
        )
        self.assertEqual(base_payload["simple_quantization"], "none")
        self.assertEqual(base_payload["inference_steps"], 50)
        self.assertEqual(base_payload["guidance_scale"], 7.0)
        self.assertEqual(base_payload["shift"], 3.0)
        self.assertFalse(base_payload["use_adg"])
        self.assertTrue(raw_payload["_meta"]["immutable"])
        self.assertEqual(
            raw_payload["_meta"]["name"],
            premium_features.DEFAULT_BASE_PRESET_NAME,
        )

    def test_ensure_default_preset_refreshes_stale_immutable_payload(self):
        """Immutable default preset file should be rewritten to current bundled defaults."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            default_dir = Path(tmp_dir) / premium_features.DEFAULT_PRESET_FOLDER
            default_dir.mkdir(parents=True, exist_ok=True)
            stale_path = default_dir / "default.json"
            stale_path.write_text(
                json.dumps(
                    {
                        "_meta": {
                            "name": premium_features.DEFAULT_PRESET_NAME,
                            "immutable": True,
                            "format": "ace_step_premium_preset",
                            "version": 1,
                        },
                        "values": {
                            "captions": "old caption",
                            "lyrics": "old lyrics",
                        },
                    }
                ),
                encoding="utf-8",
            )
            try:
                target = premium_features.ensure_default_preset()
                payload = json.loads(target.read_text(encoding="utf-8"))
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(payload["_meta"]["version"], 2)
        self.assertEqual(
            payload["values"]["captions"],
            premium_features.DEFAULT_PRESET_CAPTION,
        )
        self.assertEqual(
            payload["values"]["lyrics"],
            premium_features.DEFAULT_PRESET_LYRICS,
        )
        self.assertEqual(payload["values"]["audio_format"], "flac_mp3")
        self.assertEqual(payload["values"]["mp3_bitrate"], "320k")
        self.assertEqual(payload["values"]["simple_model_dropdown"], "acestep-v15-xl-sft")
        self.assertEqual(payload["values"]["simple_quantization"], "none")
        self.assertEqual(payload["values"]["inference_steps"], 50)
        self.assertEqual(payload["values"]["guidance_scale"], 7.0)
        self.assertEqual(payload["values"]["shift"], 3.0)
        self.assertEqual(payload["values"]["lora_path"], "")
        self.assertEqual(payload["values"]["lora_dropdown"], "")
        self.assertEqual(payload["values"]["lora_scale_slider"], 1.0)

    def test_user_preset_without_checkpoint_backfills_install_local_models_dir(self):
        """Older presets missing checkpoint info should backfill runtime default."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
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
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(payload["checkpoint_dropdown"], expected)
        self.assertEqual(payload["config_path"], "acestep-v15-xl-sft")
        self.assertEqual(payload["simple_model_dropdown"], "acestep-v15-xl-sft")

    def test_explicit_user_checkpoint_is_preserved(self):
        """Custom presets should keep user-selected checkpoint paths."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            preset_dir = Path(tmp_dir) / premium_features.USER_PRESET_FOLDER
            preset_dir.mkdir(parents=True, exist_ok=True)
            explicit = r"D:\custom\checkpoint_root"
            (preset_dir / "custom.json").write_text(
                json.dumps(
                    {
                        "values": {
                            "checkpoint_dropdown": explicit,
                            "config_path": "acestep-v15-xl-turbo",
                        }
                    }
                ),
                encoding="utf-8",
            )
            try:
                payload = premium_features.load_named_preset("custom")
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(payload["checkpoint_dropdown"], explicit)
        self.assertEqual(payload["config_path"], "acestep-v15-xl-turbo")
        self.assertEqual(payload["simple_model_dropdown"], "acestep-v15-xl-turbo")

    def test_preset_keys_track_lora_dropdown_without_use_checkbox(self):
        """Preset serialization should save LoRA selection, not removed UI toggles."""

        keys = premium_features.get_preset_component_keys()

        self.assertIn("lora_path", keys)
        self.assertIn("lora_dropdown", keys)
        self.assertIn("lora_scale_slider", keys)
        self.assertIn("simple_model_dropdown", keys)
        self.assertIn("simple_quantization", keys)
        self.assertNotIn("use_lora_checkbox", keys)

    def test_user_preset_saves_and_loads_simple_model_selector(self):
        """Create-tab model and quantization settings should persist in presets."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
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
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(remembered, "turbo fp8")
        self.assertEqual(loaded["config_path"], DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(loaded["simple_model_dropdown"], DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(loaded["quantization_checkbox"], "fp8_scaled")
        self.assertEqual(loaded["simple_quantization"], "fp8_scaled")
        self.assertEqual(
            updates[keys.index("simple_model_dropdown")].get("value"),
            DEFAULT_TURBO_DIT_MODEL,
        )
        self.assertEqual(
            updates[keys.index("simple_quantization")].get("value"),
            "fp8_scaled",
        )

    def test_user_preset_saves_and_loads_lora_selection(self):
        """LoRA path/dropdown/scale should persist through user preset save/load."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            adapter = _write_peft_adapter(Path(tmp_dir) / "Loras" / "voice")
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("lora_path")] = ""
            values[keys.index("lora_dropdown")] = str(adapter)
            values[keys.index("lora_scale_slider")] = 0.65
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("lora preset", None, *values)
                loaded = premium_features.load_named_preset("lora preset")
                updates = premium_features.load_preset_action("lora preset")
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(loaded["lora_dropdown"], str(adapter))
        self.assertEqual(loaded["lora_scale_slider"], 0.65)
        dropdown_update = updates[keys.index("lora_dropdown")]
        self.assertEqual(dropdown_update.get("value"), str(adapter))
        self.assertIn("Next run will use LoRA:", updates[len(keys)])
        self.assertTrue(updates[len(keys) + 1].get("value"))

    def test_save_rejects_system_preset_names_case_insensitive(self):
        """Users should not be able to overwrite protected system presets."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            try:
                default_updates = premium_features.save_preset_action(
                    "premium default",
                    None,
                    *values,
                )
                turbo_updates = premium_features.save_preset_action(
                    "default turbo",
                    None,
                    *values,
                )
                base_updates = premium_features.save_preset_action(
                    "default base",
                    None,
                    *values,
                )
                user_default_path = (
                    Path(tmp_dir)
                    / premium_features.USER_PRESET_FOLDER
                    / "premium default.json"
                )
                user_turbo_path = (
                    Path(tmp_dir)
                    / premium_features.USER_PRESET_FOLDER
                    / "default turbo.json"
                )
                user_base_path = (
                    Path(tmp_dir)
                    / premium_features.USER_PRESET_FOLDER
                    / "default base.json"
                )
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertIn("immutable", default_updates[1])
        self.assertIn("immutable", turbo_updates[1])
        self.assertIn("immutable", base_updates[1])
        self.assertFalse(user_default_path.exists())
        self.assertFalse(user_turbo_path.exists())
        self.assertFalse(user_base_path.exists())

    def test_startup_uses_remembered_existing_user_preset(self):
        """Startup should auto-load the last successfully saved/loaded user preset."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            keys = premium_features.get_preset_component_keys()
            values = [""] * len(keys)
            values[keys.index("config_path")] = "acestep-v15-xl-turbo"
            values[keys.index("subprocess_mode_checkbox")] = False
            try:
                premium_features.save_preset_action("daily", None, *values)
                updates = premium_features.startup_preset_updates()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        config_update = updates[keys.index("config_path")]
        simple_model_update = updates[keys.index("simple_model_dropdown")]
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(config_update.get("value"), "acestep-v15-xl-turbo")
        self.assertEqual(simple_model_update.get("value"), "acestep-v15-xl-turbo")
        self.assertEqual(dropdown_update.get("value"), "daily")
        self.assertIn("Loaded preset: daily", updates[len(keys) + 3])

    def test_startup_uses_remembered_default_turbo_preset(self):
        """Startup should remember protected system presets too."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                premium_features.set_last_used_preset_name(
                    premium_features.DEFAULT_TURBO_PRESET_NAME
                )
                updates = premium_features.startup_preset_updates()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        keys = premium_features.get_preset_component_keys()
        self.assertEqual(
            updates[keys.index("config_path")].get("value"),
            DEFAULT_TURBO_DIT_MODEL,
        )
        self.assertEqual(
            updates[keys.index("simple_model_dropdown")].get("value"),
            DEFAULT_TURBO_DIT_MODEL,
        )
        self.assertEqual(updates[keys.index("inference_steps")].get("value"), 8)
        self.assertEqual(
            updates[len(keys) + 2].get("value"),
            premium_features.DEFAULT_TURBO_PRESET_NAME,
        )

    def test_startup_uses_remembered_default_base_preset(self):
        """Startup should remember the protected XL-Base preset too."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                premium_features.set_last_used_preset_name(
                    premium_features.DEFAULT_BASE_PRESET_NAME
                )
                updates = premium_features.startup_preset_updates()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        keys = premium_features.get_preset_component_keys()
        self.assertEqual(
            updates[keys.index("config_path")].get("value"),
            DEFAULT_BASE_DIT_MODEL,
        )
        self.assertEqual(
            updates[keys.index("simple_model_dropdown")].get("value"),
            DEFAULT_BASE_DIT_MODEL,
        )
        self.assertEqual(updates[keys.index("inference_steps")].get("value"), 50)
        self.assertEqual(
            updates[len(keys) + 2].get("value"),
            premium_features.DEFAULT_BASE_PRESET_NAME,
        )

    def test_startup_falls_back_when_remembered_preset_is_missing(self):
        """Missing last-used presets should not leave the dropdown on a bad value."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                premium_features.set_last_used_preset_name("deleted preset")
                updates = premium_features.startup_preset_updates()
                remembered = premium_features.get_last_used_preset_name()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), premium_features.DEFAULT_PRESET_NAME)
        self.assertEqual(remembered, premium_features.DEFAULT_PRESET_NAME)
        self.assertIn("missing", updates[len(keys) + 3])

    def test_load_missing_preset_falls_back_to_default(self):
        """Manual loads of missing presets should resolve to the protected default."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            try:
                updates = premium_features.load_preset_action("does not exist")
                remembered = premium_features.get_last_used_preset_name()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), premium_features.DEFAULT_PRESET_NAME)
        self.assertEqual(remembered, premium_features.DEFAULT_PRESET_NAME)
        self.assertIn("missing", updates[len(keys) + 3])

    def test_corrupt_user_preset_falls_back_to_default(self):
        """Unreadable remembered presets should fall back instead of loading blanks."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            preset_dir = Path(tmp_dir) / premium_features.USER_PRESET_FOLDER
            preset_dir.mkdir(parents=True, exist_ok=True)
            (preset_dir / "broken.json").write_text("{not json", encoding="utf-8")
            try:
                premium_features.set_last_used_preset_name("broken")
                updates = premium_features.startup_preset_updates()
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        keys = premium_features.get_preset_component_keys()
        dropdown_update = updates[len(keys) + 2]
        self.assertEqual(dropdown_update.get("value"), premium_features.DEFAULT_PRESET_NAME)

    def test_user_folder_system_names_are_not_listed(self):
        """Manually created user files cannot shadow protected system presets."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.environ.get("ACESTEP_PROJECT_ROOT")
            os.environ["ACESTEP_PROJECT_ROOT"] = tmp_dir
            preset_dir = Path(tmp_dir) / premium_features.USER_PRESET_FOLDER
            preset_dir.mkdir(parents=True, exist_ok=True)
            (preset_dir / f"{premium_features.DEFAULT_PRESET_NAME}.json").write_text(
                json.dumps({"values": {"captions": "shadow"}}),
                encoding="utf-8",
            )
            (preset_dir / f"{premium_features.DEFAULT_TURBO_PRESET_NAME}.json").write_text(
                json.dumps({"values": {"config_path": "shadow"}}),
                encoding="utf-8",
            )
            (preset_dir / f"{premium_features.DEFAULT_BASE_PRESET_NAME}.json").write_text(
                json.dumps({"values": {"config_path": "shadow"}}),
                encoding="utf-8",
            )
            try:
                names = premium_features.list_preset_names()
                payload = premium_features.load_named_preset(
                    premium_features.DEFAULT_PRESET_NAME
                )
                turbo_payload = premium_features.load_named_preset(
                    premium_features.DEFAULT_TURBO_PRESET_NAME
                )
                base_payload = premium_features.load_named_preset(
                    premium_features.DEFAULT_BASE_PRESET_NAME
                )
            finally:
                if original is None:
                    os.environ.pop("ACESTEP_PROJECT_ROOT", None)
                else:
                    os.environ["ACESTEP_PROJECT_ROOT"] = original

        self.assertEqual(names.count(premium_features.DEFAULT_PRESET_NAME), 1)
        self.assertEqual(names.count(premium_features.DEFAULT_TURBO_PRESET_NAME), 1)
        self.assertEqual(names.count(premium_features.DEFAULT_BASE_PRESET_NAME), 1)
        self.assertEqual(payload["captions"], premium_features.DEFAULT_PRESET_CAPTION)
        self.assertEqual(turbo_payload["config_path"], DEFAULT_TURBO_DIT_MODEL)
        self.assertEqual(base_payload["config_path"], DEFAULT_BASE_DIT_MODEL)


if __name__ == "__main__":
    unittest.main()
