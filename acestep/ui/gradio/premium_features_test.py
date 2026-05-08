"""Tests for premium preset defaults and install-local path resolution."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from acestep.model_downloader import get_models_dir
from acestep.ui.gradio import premium_features


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


if __name__ == "__main__":
    unittest.main()
