"""Regression tests for inference metadata and seed plumbing."""

import tempfile
import unittest
from math import nan
from pathlib import Path
from unittest.mock import patch

import torch

from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS
from acestep.inference import (
    GenerationConfig,
    GenerationParams,
    _has_bounded_source_edit,
    _update_metadata_from_lm,
    generate_music,
)


class _FakeDitHandler:
    """Minimal DiT handler used to inspect generate_music kwargs."""

    lora_loaded = False
    use_lora = False
    lora_scale = 1.0

    def __init__(self, config_path: str = "", extra_outputs=None) -> None:
        self.generate_kwargs = {}
        self.prepare_seed_calls = []
        self.last_init_params = {"config_path": config_path} if config_path else {}
        self.extra_outputs = extra_outputs or {}

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
        lyrics="",
        audio_duration=None,
        src_audio=None,
        **kwargs,
    ):
        self.generate_kwargs = {
            "seed": seed,
            "use_random_seed": use_random_seed,
            "audio_code_string": audio_code_string,
            "captions": captions,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "src_audio": src_audio,
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
            "extra_outputs": {"seed_value": seed, **self.extra_outputs},
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


class BoundedSourceEditTests(unittest.TestCase):
    """Verify source-preserving edit detection for post-processing guards."""

    def test_repaint_with_valid_range_is_bounded_source_edit(self):
        """A bounded Repaint range should preserve source regions."""
        params = GenerationParams(
            task_type="repaint",
            repainting_start=40.0,
            repainting_end=50.0,
        )

        self.assertTrue(_has_bounded_source_edit(params))

    def test_full_repaint_without_end_is_not_bounded_source_edit(self):
        """Full or open-ended edits may still use global post-processing."""
        params = GenerationParams(
            task_type="repaint",
            repainting_start=0.0,
            repainting_end=-1,
        )

        self.assertFalse(_has_bounded_source_edit(params))

    def test_remix_is_not_bounded_source_edit(self):
        """Remix does not use repaint range preservation semantics."""
        params = GenerationParams(
            task_type="cover",
            repainting_start=10.0,
            repainting_end=20.0,
        )

        self.assertFalse(_has_bounded_source_edit(params))


class MetadataNormalizationTests(unittest.TestCase):
    """Verify LM metadata is sanitized before conditioning the DiT."""

    def test_generation_params_normalizes_invalid_schedule_controls(self):
        """Invalid schedule controls should not reach the diffusion sampler."""

        for bad_shift in (0.0, -1.0, nan):
            with self.subTest(shift=bad_shift):
                params = GenerationParams(
                    shift=bad_shift,
                    timesteps=[1.0, nan, 0.0],
                )

                self.assertEqual(params.shift, 1.0)
                self.assertIsNone(params.timesteps)

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

    def test_complete_lm_target_duration_uses_source_before_dit_lock(self):
        """Complete Think-mode codes should target source duration even when UI duration is auto."""

        dit_handler = _FakeDitHandler(config_path="acestep-v15-xl-base")
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            task_type="complete",
            src_audio="source.flac",
            caption="80s pop",
            lyrics="[Instrumental]",
            duration=-1,
            thinking=True,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            repainting_start=2.0,
            repainting_end=4.0,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "acestep.audio_processing.media_io.media_audio_duration_seconds",
            return_value=12.5,
        ):
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        self.assertTrue(result.success)
        self.assertEqual(llm_handler.generate_kwargs["target_duration"], 12.5)
        self.assertIsNone(dit_handler.generate_kwargs["audio_duration"])
        self.assertEqual(dit_handler.generate_kwargs["src_audio"], "source.flac")

    def test_non_vocal_lego_uses_instrumental_lyrics_before_lm_and_dit(self):
        """Non-vocal Lego should not condition LM or DiT on full vocal lyrics."""

        dit_handler = _FakeDitHandler(config_path="acestep-v15-xl-base")
        llm_handler = _FakeLlmHandler()
        params = GenerationParams(
            task_type="lego",
            instruction="Generate the GUITAR track based on the audio context:",
            src_audio="source.flac",
            caption="80s pop with prominent drums",
            lyrics="[Verse]\nSing these words",
            duration=-1,
            thinking=True,
            use_cot_metas=True,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "acestep.audio_processing.media_io.media_audio_duration_seconds",
            return_value=12.5,
        ):
            result = generate_music(
                dit_handler,
                llm_handler,
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        self.assertTrue(result.success)
        self.assertEqual("[Instrumental]", llm_handler.generate_kwargs["lyrics"])
        self.assertEqual("[Instrumental]", dit_handler.generate_kwargs["lyrics"])
        self.assertEqual("[Instrumental]", result.audios[0]["params"]["lyrics"])


class EffectiveGenerationMetadataTests(unittest.TestCase):
    """Verify saved params reflect backend task substitutions."""

    def test_lyric_repaint_saves_effective_text2music_instruction(self):
        """Lyric repaint should record the local text-to-song instruction."""

        effective_generation = {
            "requested_task_type": "repaint",
            "task_type": "text2music",
            "instruction": DEFAULT_DIT_INSTRUCTION,
            "caption": "rap Repaint the selected 1-second mask.",
            "vocal_language": "en",
            "audio_duration": 1.0,
            "repainting_start": 0.0,
            "repainting_end": 1.0,
            "lyric_repaint_local_span": True,
        }
        dit_handler = _FakeDitHandler(
            extra_outputs={"effective_generation": effective_generation}
        )
        params = GenerationParams(
            task_type="repaint",
            instruction=TASK_INSTRUCTIONS["repaint"],
            src_audio="source.wav",
            caption="rap",
            lyrics="new lyric line",
            duration=-1,
            repainting_start=1.0,
            repainting_end=2.0,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[123],
            audio_format="wav",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_music(
                dit_handler,
                _FakeLlmHandler(),
                params=params,
                config=config,
                save_dir=temp_dir,
            )

        saved_params = result.audios[0]["params"]
        self.assertEqual("repaint", saved_params["task_type"])
        self.assertEqual(DEFAULT_DIT_INSTRUCTION, saved_params["instruction"])
        self.assertEqual(TASK_INSTRUCTIONS["repaint"], saved_params["requested_instruction"])
        self.assertEqual("text2music", saved_params["effective_task_type"])
        self.assertEqual(DEFAULT_DIT_INSTRUCTION, saved_params["effective_instruction"])
        self.assertEqual(
            "rap Repaint the selected 1-second mask.",
            saved_params["effective_caption"],
        )
        self.assertTrue(saved_params["lyric_repaint_local_span"])


if __name__ == "__main__":
    unittest.main()
