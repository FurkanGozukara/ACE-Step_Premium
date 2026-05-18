"""Unit tests for model configuration and UI control settings."""

import unittest

from acestep.model_downloader import DEFAULT_TURBO_DIT_MODEL
from acestep.ui.gradio.events.generation.model_config import (
    _has_token,
    is_sft_model,
    is_pure_base_model,
    get_ui_control_config,
    get_ui_control_config_for_path,
    select_preferred_model_path,
    update_model_type_settings,
)


class HasTokenTests(unittest.TestCase):
    """Verify _has_token matches tokens at various delimiter boundaries."""

    def test_token_with_hyphens(self):
        """Standard hyphen-delimited path."""
        self.assertTrue(_has_token("sft", "acestep-sft-1b"))

    def test_token_at_start(self):
        """Token at the beginning of the string."""
        self.assertTrue(_has_token("sft", "sft-model"))

    def test_token_at_end(self):
        """Token at the end of the string."""
        self.assertTrue(_has_token("sft", "model-sft"))

    def test_token_alone(self):
        """Token is the entire string."""
        self.assertTrue(_has_token("sft", "sft"))

    def test_token_with_dots(self):
        """Dot-delimited path."""
        self.assertTrue(_has_token("sft", "model.sft.v1"))

    def test_token_with_underscores(self):
        """Underscore-delimited path."""
        self.assertTrue(_has_token("sft", "model_sft_v1"))

    def test_token_embedded_rejected(self):
        """Token inside a larger word is not matched."""
        self.assertFalse(_has_token("sft", "sftp-server"))
        self.assertFalse(_has_token("base", "database"))


class IsSftModelTests(unittest.TestCase):
    """Verify is_sft_model correctly identifies SFT model paths."""

    def test_sft_model_detected(self):
        """Paths containing 'sft' without 'turbo' should be identified as SFT."""
        self.assertTrue(is_sft_model("acestep-sft-1b-v1"))

    def test_turbo_model_not_sft(self):
        """Turbo models should not be classified as SFT even if path contains 'sft'."""
        self.assertFalse(is_sft_model("acestep-sft-turbo-1b"))

    def test_base_model_not_sft(self):
        """Plain base models should not be classified as SFT."""
        self.assertFalse(is_sft_model("acestep-base-1b"))

    def test_substring_inside_larger_word_rejected(self):
        """Word-boundary matching rejects 'sft' embedded in larger tokens.

        "sftp-server" contains "sft" but not as a delimited token.
        """
        self.assertFalse(is_sft_model("sftp-server"))

    def test_unrelated_path_not_sft(self):
        """Paths without any SFT-related substring should not match."""
        self.assertFalse(is_sft_model("acestep-v15-1b"))


class IsPureBaseModelTests(unittest.TestCase):
    """Verify is_pure_base_model correctly identifies pure base model paths."""

    def test_base_model_detected(self):
        """Paths containing 'base' without 'sft' or 'turbo' should match."""
        self.assertTrue(is_pure_base_model("acestep-base-1b"))

    def test_sft_model_not_base(self):
        """SFT models should not be classified as pure base."""
        self.assertFalse(is_pure_base_model("acestep-base-sft-1b"))

    def test_turbo_model_not_base(self):
        """Turbo models should not be classified as pure base."""
        self.assertFalse(is_pure_base_model("acestep-base-turbo-1b"))

    def test_substring_inside_larger_word_rejected(self):
        """Word-boundary matching rejects 'base' embedded in larger tokens.

        "database" contains "base" but not as a delimited token.
        """
        self.assertFalse(is_pure_base_model("database-model"))

    def test_unrelated_path_not_base(self):
        """Paths without any base-related substring should not match."""
        self.assertFalse(is_pure_base_model("acestep-v15-1b"))


class GetUiControlConfigTests(unittest.TestCase):
    """Verify get_ui_control_config returns correct defaults per model type."""

    def test_sft_model_returns_quality_defaults(self):
        """SFT models should default to the documented CFG quality schedule."""
        cfg = get_ui_control_config(is_turbo=False, is_sft=True)
        self.assertEqual(cfg["inference_steps_value"], 50)
        self.assertEqual(cfg["guidance_scale_value"], 7.0)
        self.assertFalse(cfg["use_adg_value"])
        self.assertEqual(cfg["shift_value"], 1.0)
        self.assertEqual(cfg["shift_minimum"], 1.0)
        self.assertEqual(cfg["shift_maximum"], 5.0)

    def test_base_model_returns_quality_defaults(self):
        """Base models should default to APG/CFG quality settings."""
        cfg = get_ui_control_config(is_turbo=False, is_pure_base=True)
        self.assertEqual(cfg["inference_steps_value"], 64)
        self.assertEqual(cfg["guidance_scale_value"], 7.0)
        self.assertFalse(cfg["use_adg_value"])
        self.assertEqual(cfg["shift_value"], 1.0)
        self.assertEqual(cfg["cfg_interval_start_minimum"], 0.0)
        self.assertEqual(cfg["cfg_interval_end_maximum"], 1.0)

    def test_turbo_model_returns_8_steps(self):
        """Turbo models should default to 8 inference steps."""
        cfg = get_ui_control_config(is_turbo=True)
        self.assertEqual(cfg["inference_steps_value"], 8)
        self.assertEqual(cfg["guidance_scale_value"], 1.0)
        self.assertFalse(cfg["use_adg_value"])
        self.assertEqual(cfg["shift_value"], 3.0)
        self.assertEqual(cfg["guidance_scale_maximum"], 15.0)

    def test_turbo_takes_precedence_over_sft(self):
        """When both turbo and SFT flags are set, turbo should win."""
        cfg = get_ui_control_config(is_turbo=True, is_sft=True)
        self.assertEqual(cfg["inference_steps_value"], 8)

    def test_xl_sft_path_returns_quality_steps(self):
        """The premium default XL-SFT model should use SFT defaults."""
        cfg = get_ui_control_config_for_path("acestep-v15-xl-sft")
        self.assertEqual(cfg["inference_steps_value"], 50)
        self.assertEqual(cfg["shift_value"], 1.0)
        self.assertFalse(cfg["use_adg_value"])

    def test_xl_base_path_returns_quality_steps_and_base_modes(self):
        """XL-Base should use non-turbo defaults and expose base-only modes."""
        cfg = get_ui_control_config_for_path("acestep-v15-xl-base")
        self.assertEqual(cfg["inference_steps_value"], 64)
        self.assertEqual(cfg["shift_value"], 1.0)
        self.assertFalse(cfg["use_adg_value"])
        self.assertIn("Extract", cfg["generation_mode_choices"])

    def test_xl_turbo_path_returns_8_steps(self):
        """XL-Turbo should keep the 8-step turbo default."""
        cfg = get_ui_control_config_for_path("acestep-v15-xl-turbo")
        self.assertEqual(cfg["inference_steps_value"], 8)


class UpdateModelTypeSettingsIntegrationTests(unittest.TestCase):
    """End-to-end tests: config path string in, correct step defaults out."""

    def test_sft_path_produces_quality_steps(self):
        """Passing an SFT model path should yield documented quality settings."""
        result = update_model_type_settings("acestep-v15-sft")
        # First element is the inference_steps gr.update()
        self.assertEqual(result[0]["value"], 50)
        self.assertEqual(result[2]["value"], False)
        self.assertEqual(result[3]["value"], 1.0)
        self.assertEqual(result[3]["minimum"], 1.0)
        self.assertEqual(result[3]["maximum"], 5.0)

    def test_turbo_path_produces_8_steps(self):
        """Passing a turbo model path should yield 8 inference steps."""
        result = update_model_type_settings("acestep-v15-turbo")
        self.assertEqual(result[0]["value"], 8)
        self.assertEqual(result[1]["value"], 1.0)
        self.assertFalse(result[1]["visible"])

    def test_base_path_produces_quality_steps(self):
        """Passing a base model path should yield high-quality inference steps."""
        result = update_model_type_settings("acestep-v15-base")
        self.assertEqual(result[0]["value"], 64)
        self.assertEqual(result[1]["value"], 7.0)
        self.assertTrue(result[1]["visible"])
        self.assertFalse(result[2]["value"])
        self.assertEqual(result[3]["value"], 1.0)

    def test_none_path_does_not_crash(self):
        """Passing None as config_path should not raise."""
        result = update_model_type_settings(None)
        self.assertEqual(result[0]["value"], 50)

    def test_substring_no_false_positive_end_to_end(self):
        """Word-boundary matching prevents false positives end-to-end.

        "sftp-server" contains "sft" but not as a delimited token,
        so it correctly falls through to the non-turbo SFT-compatible default.
        """
        result = update_model_type_settings("sftp-server")
        self.assertEqual(result[0]["value"], 50)


class PreferredModelPathTests(unittest.TestCase):
    """Verify app startup selects the same fresh-install model everywhere."""

    def test_prefers_xl_turbo(self):
        models = ["acestep-v15-xl-turbo", "acestep-v15-xl-sft"]
        self.assertEqual(select_preferred_model_path(models), "acestep-v15-xl-turbo")

    def test_falls_back_to_first_available_model(self):
        models = ["custom-model-a", "custom-model-b"]
        self.assertEqual(select_preferred_model_path(models), "custom-model-a")

    def test_empty_models_falls_back_to_xl_turbo(self):
        self.assertEqual(select_preferred_model_path([]), DEFAULT_TURBO_DIT_MODEL)


if __name__ == "__main__":
    unittest.main()
