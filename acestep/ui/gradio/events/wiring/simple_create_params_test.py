"""Tests for simple Create tab parameter mapping."""

import unittest
import re

from acestep.ui.gradio.events.wiring.simple_create_params import (
    prepare_simple_generation,
)


INFERENCE_STEPS_INDEX = 27
GUIDANCE_SCALE_INDEX = 28
USE_ADG_INDEX = 29
SHIFT_INDEX = 30
CONFIG_PATH_INDEX = 33
DCW_ENABLED_INDEX = 34
DCW_MODE_INDEX = 35
DCW_SCALER_INDEX = 36
DCW_HIGH_SCALER_INDEX = 37
INFER_METHOD_INDEX = 38
SAMPLER_MODE_INDEX = 39
VELOCITY_NORM_THRESHOLD_INDEX = 40
VELOCITY_EMA_FACTOR_INDEX = 41
CUSTOM_TIMESTEPS_INDEX = 42
DCW_WAVELET_INDEX = 43


class SimpleCreateParamsTests(unittest.TestCase):
    """Verify simple Create tab values map to the full generation contract."""

    def test_turbo_auto_duration_enables_lm_think_and_cot_metadata(self):
        """Turbo auto duration should use the LM path so duration can follow lyrics."""

        result = prepare_simple_generation(
            caption="emotional pop",
            lyrics="verse\nchorus",
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=-1,
            batch_size=2,
            random_seed=True,
            seed="-1",
            quantization="fp8_scaled",
            model_path="acestep-v15-xl-turbo",
        )

        self.assertEqual(result[5], -1.0)
        self.assertEqual(result[6], 2)
        self.assertEqual(result[7], "fp8_scaled")
        self.assertTrue(result[8])
        self.assertTrue(result[9])
        self.assertTrue(result[10])
        self.assertEqual(result[12], "-1")
        self.assertTrue(result[18])
        self.assertTrue(result[19])
        self.assertTrue(result[20])
        self.assertTrue(result[21])
        self.assertIn("Auto duration", result[26])
        self.assertEqual(result[INFERENCE_STEPS_INDEX], 8)
        self.assertEqual(result[GUIDANCE_SCALE_INDEX], 1.0)
        self.assertEqual(result[CONFIG_PATH_INDEX], "acestep-v15-xl-turbo")
        self.assertTrue(result[DCW_ENABLED_INDEX])
        self.assertEqual(result[DCW_MODE_INDEX], "double")
        self.assertEqual(result[DCW_SCALER_INDEX], 0.02)
        self.assertEqual(result[DCW_HIGH_SCALER_INDEX], 0.06)
        self.assertEqual(result[INFER_METHOD_INDEX], "ode")
        self.assertEqual(result[SAMPLER_MODE_INDEX], "euler")
        self.assertEqual(result[VELOCITY_NORM_THRESHOLD_INDEX], 0.0)
        self.assertEqual(result[VELOCITY_EMA_FACTOR_INDEX], 0.0)
        self.assertEqual(result[CUSTOM_TIMESTEPS_INDEX], "")
        self.assertEqual(result[DCW_WAVELET_INDEX], "haar")

    def test_base_model_selector_is_forwarded_to_advanced_generation(self):
        """The simple Base selection should apply quality defaults."""

        result = prepare_simple_generation(
            caption="emotional pop",
            lyrics="verse",
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=60,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
            model_path="acestep-v15-xl-base",
        )

        self.assertEqual(result[INFERENCE_STEPS_INDEX], 64)
        self.assertEqual(result[GUIDANCE_SCALE_INDEX], 7.0)
        self.assertFalse(result[USE_ADG_INDEX])
        self.assertEqual(result[SHIFT_INDEX], 1.0)
        self.assertEqual(result[CONFIG_PATH_INDEX], "acestep-v15-xl-base")
        self.assertFalse(result[8])
        self.assertFalse(result[9])
        self.assertFalse(result[10])
        self.assertFalse(result[19])
        self.assertFalse(result[20])
        self.assertFalse(result[21])
        self.assertFalse(result[DCW_ENABLED_INDEX])
        self.assertEqual(result[DCW_MODE_INDEX], "double")
        self.assertEqual(result[DCW_SCALER_INDEX], 0.0)
        self.assertEqual(result[DCW_HIGH_SCALER_INDEX], 0.0)
        self.assertEqual(result[INFER_METHOD_INDEX], "ode")
        self.assertEqual(result[SAMPLER_MODE_INDEX], "euler")
        self.assertEqual(result[VELOCITY_NORM_THRESHOLD_INDEX], 0.0)
        self.assertEqual(result[VELOCITY_EMA_FACTOR_INDEX], 0.0)
        self.assertEqual(result[CUSTOM_TIMESTEPS_INDEX], "")
        self.assertEqual(result[DCW_WAVELET_INDEX], "haar")

    def test_sft_model_selector_applies_quality_defaults(self):
        """The simple SFT selection should apply LM-assisted SFT defaults."""

        result = prepare_simple_generation(
            caption="emotional pop",
            lyrics="verse",
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=60,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
            model_path="acestep-v15-xl-sft",
        )

        self.assertEqual(result[INFERENCE_STEPS_INDEX], 50)
        self.assertEqual(result[GUIDANCE_SCALE_INDEX], 7.0)
        self.assertFalse(result[USE_ADG_INDEX])
        self.assertEqual(result[SHIFT_INDEX], 1.0)
        self.assertEqual(result[CONFIG_PATH_INDEX], "acestep-v15-xl-sft")
        self.assertTrue(result[8])
        self.assertTrue(result[9])
        self.assertFalse(result[10])
        self.assertTrue(result[19])
        self.assertTrue(result[20])
        self.assertTrue(result[21])
        self.assertFalse(result[DCW_ENABLED_INDEX])

    def test_sft_auto_duration_uses_lm_planning(self):
        """SFT auto duration should keep LM/Think metadata planning enabled."""

        result = prepare_simple_generation(
            caption="conscious melodic rap with 808 bass",
            lyrics=" ".join(["word"] * 480),
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=-1,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
            model_path="acestep-v15-xl-sft",
        )

        self.assertEqual(result[5], -1.0)
        self.assertTrue(result[8])
        self.assertTrue(result[9])
        self.assertFalse(result[10])
        self.assertTrue(result[18])
        self.assertTrue(result[19])
        self.assertTrue(result[20])
        self.assertTrue(result[21])
        self.assertIn("Auto duration", result[26])
        self.assertFalse(result[DCW_ENABLED_INDEX])
        self.assertEqual(result[DCW_SCALER_INDEX], 0.0)
        self.assertEqual(result[DCW_HIGH_SCALER_INDEX], 0.0)

    def test_fixed_duration_keeps_seconds_and_quality_path(self):
        """A positive duration should keep fixed seconds without disabling quality defaults."""

        result = prepare_simple_generation(
            caption="cinematic instrumental",
            lyrics="ignored",
            vocal_language="en",
            instrumental=True,
            vocal_gender="female",
            duration=90,
            batch_size=None,
            random_seed=True,
            seed="",
            quantization=None,
        )

        self.assertEqual(result[3], "")
        self.assertEqual(result[4], "unknown")
        self.assertEqual(result[5], 90.0)
        self.assertEqual(result[6], 1)
        self.assertEqual(result[7], "none")
        self.assertTrue(result[8])
        self.assertTrue(result[9])
        self.assertTrue(result[10])
        self.assertTrue(result[11])
        self.assertEqual(result[12], "-1")
        self.assertFalse(result[18])
        self.assertTrue(result[19])
        self.assertTrue(result[20])
        self.assertTrue(result[21])
        self.assertIn("Fixed duration: 90s", result[26])

    def test_vocal_gender_rewrites_existing_prompt_gender(self):
        """Female selection should rewrite an existing male vocal direction."""

        result = prepare_simple_generation(
            caption="modern pop, soulful confident male vocal, polished mix",
            lyrics="verse",
            vocal_language="en",
            instrumental=False,
            vocal_gender="female",
            duration=60,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
        )

        self.assertIn("female vocal", result[2])
        self.assertIsNone(re.search(r"\bmale vocal\b", result[2]))

    def test_vocal_gender_appends_when_prompt_has_no_gender(self):
        """Gender selection should still guide prompts without a vocal tag."""

        result = prepare_simple_generation(
            caption="cinematic pop anthem",
            lyrics="verse",
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=60,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
        )

        self.assertEqual(result[2], "cinematic pop anthem, male vocal.")

    def test_instrumental_removes_gendered_vocal_direction(self):
        """Instrumental mode should not keep a conflicting vocal-gender prompt."""

        result = prepare_simple_generation(
            caption="cinematic pop, soulful confident female vocal, warm drums",
            lyrics="ignored",
            vocal_language="en",
            instrumental=True,
            vocal_gender="female",
            duration=60,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
        )

        self.assertIn("instrumental arrangement, no vocals", result[2])
        self.assertNotIn("female vocal", result[2])

    def test_formatted_metadata_is_reused_for_simple_generation(self):
        """Metadata returned by enhancement should feed the advanced contract."""

        result = prepare_simple_generation(
            caption="cinematic pop",
            lyrics="verse",
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=95,
            batch_size=1,
            random_seed=True,
            seed="-1",
            quantization="none",
            formatted_bpm=118,
            formatted_key_scale="C Major",
            formatted_time_signature="4/4",
            is_format_caption=True,
        )

        self.assertFalse(result[14])
        self.assertFalse(result[15])
        self.assertFalse(result[16])
        self.assertEqual(result[22], 118)
        self.assertEqual(result[23], "C Major")
        self.assertEqual(result[24], "4/4")
        self.assertTrue(result[25])

    def test_fixed_seed_is_passed_to_generation_contract(self):
        """Simple mode should support reproducible fixed seeds."""

        result = prepare_simple_generation(
            caption="cinematic pop",
            lyrics="verse",
            vocal_language="en",
            instrumental=False,
            vocal_gender="male",
            duration=60,
            batch_size=1,
            random_seed=False,
            seed="12345",
            quantization="none",
        )

        self.assertFalse(result[11])
        self.assertEqual(result[12], "12345")


if __name__ == "__main__":
    unittest.main()
