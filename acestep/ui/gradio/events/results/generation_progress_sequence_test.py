"""Tests for sequential Gradio Songs generation behavior."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acestep.audio_processing.auto_editor_trim import SilenceTrimResult
from acestep.ui.gradio.events.generation.generation_count import (
    normalize_generation_count,
    seed_for_generation_index,
)
from acestep.ui.gradio.events.results import generation_progress


def _progress_args(**overrides):
    """Return minimal valid ``generate_with_progress`` keyword arguments."""

    args = {
        "captions": "caption", "lyrics": "lyrics", "bpm": None,
        "key_scale": "", "time_signature": "", "vocal_language": "en",
        "inference_steps": 8, "guidance_scale": 1.0,
        "random_seed_checkbox": False, "seed": "100",
        "reference_audio": None, "audio_duration": -1, "batch_size_input": 1,
        "src_audio": None, "text2music_audio_code_string": "",
        "repainting_start": 0.0, "repainting_end": -1,
        "instruction_display_gen": "", "audio_cover_strength": 1.0,
        "cover_noise_strength": 0.0, "task_type": "text2music",
        "no_fsq": False, "use_adg": False,
        "cfg_interval_start": 0.0, "cfg_interval_end": 1.0, "shift": 3.0,
        "infer_method": "ode", "sampler_mode": "euler",
        "velocity_norm_threshold": 0.0, "velocity_ema_factor": 0.0,
        "dcw_enabled": True, "dcw_mode": "double", "dcw_scaler": 0.02,
        "dcw_high_scaler": 0.06, "dcw_wavelet": "haar",
        "custom_timesteps": "", "audio_format": "flac",
        "mp3_bitrate": "320k", "mp3_sample_rate": 48000,
        "lm_temperature": 0.85, "think_checkbox": False, "lm_cfg_scale": 2.0,
        "lm_top_k": 0, "lm_top_p": 0.9,
        "lm_negative_prompt": "NO USER INPUT",
        "use_cot_metas": False, "use_cot_caption": False,
        "use_cot_language": False, "is_format_caption": False,
        "constrained_decoding_debug": False, "allow_lm_batch": True,
        "auto_score": False, "auto_lrc": False, "score_scale": 0.5,
        "lm_batch_chunk_size": 8, "enable_normalization": True,
        "normalization_db": -1.0, "fade_in_duration": 0.0,
        "fade_out_duration": 0.0, "latent_shift": 0.0, "latent_rescale": 1.0,
        "repaint_mode": "balanced", "repaint_strength": 0.5,
        "retake_variance": 0.0, "retake_seed": "",
        "flow_edit_morph": False, "flow_edit_source_caption": "",
        "flow_edit_source_lyrics": "", "flow_edit_n_min": 0.0,
        "flow_edit_n_max": 1.0, "flow_edit_n_avg": 1,
        "progress": lambda *_args, **_kwargs: None,
    }
    args.update(overrides)
    return args


def _fake_result(seed_value: str):
    """Return a minimal successful generation result for one song."""

    return SimpleNamespace(
        success=True,
        status_message="ok",
        audios=[
            {
                "key": f"song-{seed_value}",
                "tensor": object(),
                "sample_rate": 48000,
                "params": {"audio_codes": f"code-{seed_value}"},
            }
        ],
        extra_outputs={
            "seed_value": seed_value,
            "time_costs": {"dit_total_time_cost": 1.0},
            "lm_metadata": {"seed": seed_value},
        },
    )


def _fake_tensor_result(seed_value: str):
    """Return a successful generation result with trimmable tensor audio."""

    import torch

    tensor = torch.zeros(1, 8)
    tensor[:, 2:6] = 0.25
    result = _fake_result(seed_value)
    result.audios[0]["tensor"] = tensor
    return result


class SequentialGenerationCountTests(unittest.TestCase):
    """Verify the Songs control drives sequential one-song generations."""

    def test_seed_helper_increments_fixed_seed_without_count_cap(self):
        """The helper should increment fixed seeds without capping song count."""

        self.assertEqual(normalize_generation_count(99), 99)
        self.assertEqual(seed_for_generation_index("100", 2, random_seed=False), [102])
        self.assertIsNone(seed_for_generation_index("100", 2, random_seed=True))

    def test_generate_with_progress_loops_with_fixed_seed_increment(self):
        """Multiple Songs should call the backend once per song with seed + index."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            calls.append(config)
            return _fake_result(str(config.seeds[0]))

        with tempfile.TemporaryDirectory() as tmp:
            final = self._run_generation(
                tmp,
                fake_generate_music,
                batch_size_input=3,
                random_seed_checkbox=False,
                seed="100",
            )

        self.assertEqual([call.batch_size for call in calls], [1, 1, 1])
        self.assertEqual([call.allow_lm_batch for call in calls], [False, False, False])
        self.assertEqual([call.seeds for call in calls], [[100], [101], [102]])
        self.assertEqual(final[11], "100, 101, 102")
        self.assertEqual(final[47][:3], ["code-100", "code-101", "code-102"])

    def test_generate_with_progress_saves_outputs_beyond_visible_slots(self):
        """Songs above eight should still save files and expose all codes."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            calls.append(config)
            return _fake_result(str(config.seeds[0]))

        with tempfile.TemporaryDirectory() as tmp:
            final = self._run_generation(
                tmp,
                fake_generate_music,
                batch_size_input=10,
                random_seed_checkbox=False,
                seed="200",
            )

        self.assertEqual(len(calls), 10)
        self.assertEqual([call.batch_size for call in calls], [1] * 10)
        self.assertEqual(final[47][:10], [f"code-{seed}" for seed in range(200, 210)])
        self.assertIn("song-209.flac", "\n".join(final[8]))
        self.assertIn("song-209.json", "\n".join(final[8]))

    def test_generate_with_progress_keeps_random_seed_mode_random_per_song(self):
        """Random seed mode should not pass fixed seeds into backend calls."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            calls.append(config)
            return _fake_result(str(900 + len(calls)))

        with tempfile.TemporaryDirectory() as tmp:
            final = self._run_generation(
                tmp,
                fake_generate_music,
                batch_size_input=2,
                random_seed_checkbox=True,
                seed="100",
            )

        self.assertEqual([call.batch_size for call in calls], [1, 1])
        self.assertEqual([call.use_random_seed for call in calls], [True, True])
        self.assertEqual([call.seeds for call in calls], [None, None])
        self.assertEqual(final[11], "901, 902")
        self.assertEqual(final[47][:2], ["code-901", "code-902"])

    def test_extract_trim_shortens_saved_tensor_and_metadata(self):
        """Extract trim should modify saved audio tensors and sidecar metadata."""

        saved_shapes = []
        json_payloads = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            return _fake_tensor_result(str(config.seeds[0]))

        def fake_save_audio(audio_data, output_path, **_kwargs):
            saved_shapes.append(tuple(audio_data.shape))
            return output_path

        def fake_write_json(_path, payload):
            json_payloads.append(payload)
            return str(_path)

        def fake_trim(audio_tensor, **_kwargs):
            metadata = {
                "enabled": True,
                "applied": True,
                "reason": "auto_editor_trimmed",
                "segments": [{"start_sample": 2, "end_sample": 6}],
            }
            return SilenceTrimResult(audio_tensor[:, 2:6], metadata)

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "acestep.ui.gradio.events.results.generation_progress.trim_silent_edges",
                side_effect=fake_trim,
            ):
                self._run_generation(
                    tmp,
                    fake_generate_music,
                    task_type="extract",
                    extract_trim_empty_output=True,
                    extract_trim_threshold_db=-40.0,
                    save_audio_side_effect=fake_save_audio,
                    write_json_side_effect=fake_write_json,
                )

        self.assertEqual([(1, 4)], saved_shapes)
        sample_sidecar = next(payload for payload in json_payloads if "extract_trim" in payload)
        self.assertTrue(sample_sidecar["extract_trim"]["applied"])
        self.assertEqual(2, sample_sidecar["extract_trim"]["segments"][0]["start_sample"])
        self.assertEqual(6, sample_sidecar["extract_trim"]["segments"][0]["end_sample"])

    def _run_generation(
        self,
        tmp: str,
        fake_generate_music,
        save_audio_side_effect=None,
        write_json_side_effect=None,
        **overrides,
    ):
        """Run ``generate_with_progress`` with filesystem and audio writes mocked."""

        gpu_config = SimpleNamespace(
            save_memory_mode=False,
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            gpu_memory_gb=24.0,
        )
        save_side_effect = save_audio_side_effect or (lambda output_path, **_kwargs: output_path)
        write_json_effect = write_json_side_effect or (lambda *_args, **_kwargs: None)
        patches = [
            patch.object(generation_progress, "get_global_gpu_config", return_value=gpu_config),
            patch.object(generation_progress, "check_duration_limit", return_value=(True, "")),
            patch.object(generation_progress, "create_generation_run_dir", return_value=Path(tmp)),
            patch.object(generation_progress, "persist_generation_inputs", return_value={}),
            patch.object(generation_progress, "build_generation_manifest", return_value=str(Path(tmp) / "manifest.json")),
            patch.object(generation_progress, "write_json", side_effect=write_json_effect),
            patch("acestep.audio_utils.save_audio", side_effect=save_side_effect),
            patch("acestep.inference.generate_music", side_effect=fake_generate_music),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            dit_handler = SimpleNamespace(last_init_params={"config_path": "acestep-v15-xl-turbo"})
            llm_handler = SimpleNamespace(llm_initialized=False, last_init_params={})
            outputs = list(
                generation_progress.generate_with_progress(
                    dit_handler,
                    llm_handler,
                    **_progress_args(**overrides),
                )
            )
        return outputs[-1]


if __name__ == "__main__":
    unittest.main()
