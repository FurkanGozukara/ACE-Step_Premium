"""Regression tests for inference metadata and seed plumbing."""

import tempfile
import unittest
from pathlib import Path

import torch

from acestep.inference import (
    GenerationConfig,
    GenerationParams,
    _update_metadata_from_lm,
    generate_music,
)


class _FakeDitHandler:
    """Minimal DiT handler used to inspect generate_music kwargs."""

    lora_loaded = False
    use_lora = False
    lora_scale = 1.0

    def __init__(self, config_path: str = "") -> None:
        self.generate_kwargs = {}
        self.prepare_seed_calls = []
        self.last_init_params = {"config_path": config_path} if config_path else {}

    def prepare_seeds(self, actual_batch_size, seed, use_random_seed):
        self.prepare_seed_calls.append((actual_batch_size, seed, use_random_seed))
        if use_random_seed:
            seeds = [111 + idx for idx in range(actual_batch_size)]
        else:
            values = [part.strip() for part in str(seed).split(",") if part.strip()]
            seeds = [int(value) for value in values[:actual_batch_size]]
        return seeds, ", ".join(str(seed_value) for seed_value in seeds)

    def generate_music(
        self,
        seed=None,
        use_random_seed=True,
        audio_code_string="",
        captions="",
        **kwargs,
    ):
        self.generate_kwargs = {
            "seed": seed,
            "use_random_seed": use_random_seed,
            "audio_code_string": audio_code_string,
            "captions": captions,
            **kwargs,
        }
        return {
            "success": True,
            "status_message": "ok",
            "audios": [
                {
                    "tensor": torch.zeros(2, 4800),
                    "sample_rate": 48000,
                }
            ],
            "extra_outputs": {"seed_value": seed},
        }


class _FakeLlmHandler:
    """Minimal initialized LM handler used to inspect LM kwargs."""

    llm_initialized = True

    def __init__(self) -> None:
        self.generate_kwargs = {}

    def generate_with_stop_condition(self, **kwargs):
        self.generate_kwargs = kwargs
        return {
            "success": True,
            "metadata": {
                "bpm": 40,
                "duration": 10,
                "keyscale": "C major",
                "timesignature": "4",
                "vocal_language": "en",
            },
            "audio_codes": "<|audio_code_1|><|audio_code_2|>",
            "extra_outputs": {"time_costs": {}},
        }


class MetadataNormalizationTests(unittest.TestCase):
    """Verify LM metadata is sanitized before conditioning the DiT."""

    def test_low_lm_bpm_is_normalized_to_musical_full_time(self):
        """A 40 BPM LM guess should condition generation as 80 BPM."""

        bpm, *_ = _update_metadata_from_lm(
            metadata={"bpm": 40},
            bpm=None,
            key_scale="",
            time_signature="",
            audio_duration=None,
            vocal_language="en",
            caption="",
            lyrics="",
        )

        self.assertEqual(bpm, 80)

    def test_rap_half_time_lm_bpm_is_normalized_to_full_time(self):
        """A 67 BPM rap metadata guess should condition as 134 BPM."""

        bpm, *_ = _update_metadata_from_lm(
            metadata={"bpm": 67},
            bpm=None,
            key_scale="",
            time_signature="",
            audio_duration=None,
            vocal_language="en",
            caption="conscious melodic rap with 808 bass",
            lyrics="[verse]\nwords",
        )

        self.assertEqual(bpm, 134)

    def test_long_rap_lm_duration_is_raised_to_lyric_density_floor(self):
        """A long rap lyric should not be squeezed into an underfit duration."""

        lyrics = " ".join(["word"] * 480)
        _, _, _, duration, *_ = _update_metadata_from_lm(
            metadata={"duration": 156},
            bpm=120,
            key_scale="",
            time_signature="",
            audio_duration=-1,
            vocal_language="en",
            caption="melodic rap",
            lyrics=lyrics,
        )

        self.assertEqual(duration, 205.0)


class SeedPlumbingTests(unittest.TestCase):
    """Verify random seed mode uses one resolved seed list end-to-end."""

    def test_random_seed_is_resolved_once_for_lm_and_dit(self):
        """LM semantic codes and DiT generation should share the same seed."""

        dit_handler = _FakeDitHandler()
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            caption="melodic rap",
            lyrics="[verse]\nwords",
            duration=-1,
            thinking=True,
            use_cot_metas=True,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=True,
            seeds=None,
            audio_format="wav",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )
            self.assertTrue(Path(result.audios[0]["path"]).exists())

        self.assertTrue(result.success)
        self.assertEqual(llm_handler.generate_kwargs["seeds"], [111])
        self.assertEqual(dit_handler.generate_kwargs["seed"], "111")
        self.assertFalse(dit_handler.generate_kwargs["use_random_seed"])
        self.assertEqual(result.audios[0]["params"]["seed"], 111)
        self.assertEqual(result.audios[0]["params"]["cot_bpm"], 80)


class LmAudioCodeRoutingTests(unittest.TestCase):
    """Verify Think mode only injects LM semantic codes into compatible DiT paths."""

    def test_sft_text2music_uses_lm_metadata_without_audio_code_hints(self):
        """SFT should not turn text2music into cover-style code conditioning."""

        dit_handler = _FakeDitHandler(config_path="acestep-v15-xl-sft")
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            caption="conscious melodic rap",
            lyrics="[verse]\nwords",
            duration=-1,
            thinking=True,
            use_cot_metas=True,
            use_cot_caption=True,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        self.assertTrue(result.success)
        self.assertEqual(llm_handler.generate_kwargs["infer_type"], "dit")
        self.assertEqual(dit_handler.generate_kwargs["audio_code_string"], "")
        self.assertEqual(dit_handler.generate_kwargs["captions"], "conscious melodic rap")
        self.assertEqual(result.audios[0]["params"]["audio_codes"], "")

    def test_turbo_text2music_keeps_lm_audio_code_hints(self):
        """Turbo should keep the lm_dit semantic-code path."""

        dit_handler = _FakeDitHandler(config_path="acestep-v15-xl-turbo")
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            caption="conscious melodic rap",
            lyrics="[verse]\nwords",
            duration=-1,
            thinking=True,
            use_cot_metas=True,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        expected_codes = "<|audio_code_1|><|audio_code_2|>"
        self.assertTrue(result.success)
        self.assertEqual(llm_handler.generate_kwargs["infer_type"], "llm_dit")
        self.assertEqual(dit_handler.generate_kwargs["audio_code_string"], expected_codes)
        self.assertEqual(result.audios[0]["params"]["audio_codes"], expected_codes)

    def test_sft_text2music_can_force_lm_audio_code_hints(self):
        """The explicit toggle should allow original-style code hints on SFT."""

        dit_handler = _FakeDitHandler(config_path="acestep-v15-xl-sft")
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            caption="conscious melodic rap",
            lyrics="[verse]\nwords",
            duration=-1,
            thinking=True,
            generate_lm_audio_codes=True,
            use_cot_metas=True,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        expected_codes = "<|audio_code_1|><|audio_code_2|>"
        self.assertTrue(result.success)
        self.assertEqual(llm_handler.generate_kwargs["infer_type"], "llm_dit")
        self.assertEqual(dit_handler.generate_kwargs["audio_code_string"], expected_codes)

    def test_turbo_text2music_can_disable_lm_audio_code_hints(self):
        """The explicit toggle should allow turbo A/B tests without code hints."""

        dit_handler = _FakeDitHandler(config_path="acestep-v15-xl-turbo")
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            caption="conscious melodic rap",
            lyrics="[verse]\nwords",
            duration=-1,
            thinking=True,
            generate_lm_audio_codes=False,
            use_cot_metas=True,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        self.assertTrue(result.success)
        self.assertEqual(llm_handler.generate_kwargs["infer_type"], "dit")
        self.assertEqual(dit_handler.generate_kwargs["audio_code_string"], "")
        self.assertEqual(result.audios[0]["params"]["audio_codes"], "")


if __name__ == "__main__":
    unittest.main()
