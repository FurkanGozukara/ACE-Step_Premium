"""Tests for simple Create tab parameter mapping."""

import unittest
import re

from acestep.ui.gradio.events.wiring.simple_create_params import (
    prepare_simple_generation,
)


class SimpleCreateParamsTests(unittest.TestCase):
    """Verify simple Create tab values map to the full generation contract."""

    def test_auto_duration_enables_lm_think_and_cot_metadata(self):
        """Auto duration should use the LM path so duration can follow lyrics."""

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
        )

        self.assertEqual(result[5], -1.0)
        self.assertEqual(result[6], 2)
        self.assertEqual(result[7], "fp8_scaled")
        self.assertTrue(result[8])
        self.assertTrue(result[9])
        self.assertTrue(result[10])
        self.assertEqual(result[11], "-1")
        self.assertTrue(result[17])
        self.assertTrue(result[18])
        self.assertTrue(result[19])
        self.assertTrue(result[20])
        self.assertIn("Auto duration", result[25])

    def test_fixed_duration_uses_direct_generation_path(self):
        """A positive duration should be treated as explicit fixed seconds."""

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
        self.assertFalse(result[8])
        self.assertFalse(result[9])
        self.assertTrue(result[10])
        self.assertEqual(result[11], "-1")
        self.assertFalse(result[17])
        self.assertFalse(result[18])
        self.assertFalse(result[19])
        self.assertFalse(result[20])
        self.assertIn("Fixed duration: 90s", result[25])

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

        self.assertFalse(result[13])
        self.assertFalse(result[14])
        self.assertFalse(result[15])
        self.assertEqual(result[21], 118)
        self.assertEqual(result[22], "C Major")
        self.assertEqual(result[23], "4/4")
        self.assertTrue(result[24])

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

        self.assertFalse(result[10])
        self.assertEqual(result[11], "12345")


if __name__ == "__main__":
    unittest.main()
