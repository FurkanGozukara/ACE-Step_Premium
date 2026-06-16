"""Tests for sequential Gradio Songs generation behavior."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acestep.audio_processing.auto_editor_trim import SilenceTrimResult
from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS, TRACK_NAMES
from acestep.ui.gradio.events.generation.generation_count import (
    normalize_generation_count,
    seed_for_generation_index,
)
from acestep.ui.gradio.events.generation.strength_defaults import (
    DEFAULT_AUDIO_COVER_STRENGTH,
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
        "instruction_display_gen": "",
        "audio_cover_strength": DEFAULT_AUDIO_COVER_STRENGTH,
        "cover_noise_strength": 0.0, "task_type": "text2music",
        "no_fsq": False, "use_adg": False,
        "cfg_interval_start": 0.0, "cfg_interval_end": 1.0, "shift": 3.0,
        "infer_method": "ode", "sampler_mode": "euler",
        "velocity_norm_threshold": 0.0, "velocity_ema_factor": 0.0,
        "dcw_enabled": True, "dcw_mode": "double", "dcw_scaler": 0.02,
        "dcw_high_scaler": 0.06, "dcw_wavelet": "haar",
        "custom_timesteps": "", "audio_format": "mp3",
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
        self.assertEqual(final[19], "100, 101, 102")
        self.assertEqual(final[55][:3], ["code-100", "code-101", "code-102"])

    def test_instrumental_checkbox_sets_generation_params(self):
        """Instrumental checkbox should reach backend params as instrumental."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            calls.append(params)
            return _fake_result(str(config.seeds[0]))

        with tempfile.TemporaryDirectory() as tmp:
            self._run_generation(
                tmp,
                fake_generate_music,
                task_type="repaint",
                src_audio="source.wav",
                lyrics="new lyric line",
                vocal_language="en",
                instrumental_checkbox=True,
                repaint_dont_switch_with_lyrics=True,
            )

        self.assertEqual(1, len(calls))
        params = calls[0]
        self.assertTrue(params.instrumental)
        self.assertEqual("[Instrumental]", params.lyrics)
        self.assertEqual("unknown", params.vocal_language)
        self.assertTrue(params.repaint_dont_switch_with_lyrics)

    def test_instrumental_repaint_does_not_force_lyric_repaint_switch(self):
        """Instrumental Repaint should stay instrumental without enabling lyric opt-out."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            calls.append(params)
            return _fake_result(str(config.seeds[0]))

        with tempfile.TemporaryDirectory() as tmp:
            self._run_generation(
                tmp,
                fake_generate_music,
                task_type="repaint",
                src_audio="source.wav",
                lyrics="",
                vocal_language="en",
                instrumental_checkbox=True,
                repaint_dont_switch_with_lyrics=False,
            )

        self.assertEqual(1, len(calls))
        params = calls[0]
        self.assertTrue(params.instrumental)
        self.assertEqual("[Instrumental]", params.lyrics)
        self.assertEqual("unknown", params.vocal_language)
        self.assertFalse(params.repaint_dont_switch_with_lyrics)

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
        self.assertEqual(final[55][:10], [f"code-{seed}" for seed in range(200, 210)])
        self.assertIn("song-209.mp3", "\n".join(final[16]))
        self.assertIn("song-209.json", "\n".join(final[16]))

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
        self.assertEqual(final[19], "901, 902")
        self.assertEqual(final[55][:2], ["code-901", "code-902"])

    def test_extract_all_stems_generates_each_stem_with_suffixes(self):
        """Extract-all-stems should call the backend once per supported stem."""

        backend_captions = []
        backend_instructions = []
        backend_seeds = []
        events = []
        saved_paths = []
        json_payloads = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            backend_captions.append(params.caption)
            backend_instructions.append(params.instruction)
            backend_seeds.append(config.seeds)
            events.append(f"backend:{params.caption}")
            return _fake_tensor_result(str(config.seeds[0]))

        def fake_save_audio(audio_data, output_path, **_kwargs):
            _ = audio_data
            saved_paths.append(output_path)
            events.append(f"save:{Path(output_path).stem}")
            return output_path

        def fake_write_json(_path, payload):
            json_payloads.append(payload)
            return str(_path)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            generation_progress,
            "save_extract_remaining_audio",
            return_value={"applied": False},
        ):
            final = self._run_generation(
                tmp,
                fake_generate_music,
                task_type="extract",
                src_audio=r"C:\music\Alpha Song.mp3",
                track_name=None,
                extract_all_stems=True,
                audio_format="wav",
                extract_trim_empty_output=False,
                save_audio_side_effect=fake_save_audio,
                write_json_side_effect=fake_write_json,
            )

        self.assertEqual(TRACK_NAMES, backend_captions)
        expected_instructions = [
            f"Extract the {stem.upper()} track from the audio:" for stem in TRACK_NAMES
        ]
        self.assertEqual(expected_instructions, backend_instructions)
        self.assertEqual([[100]] * len(TRACK_NAMES), backend_seeds)
        self.assertEqual(len(TRACK_NAMES), len(saved_paths))
        expected_events = []
        for stem in TRACK_NAMES:
            suffix = "vocal" if stem == "vocals" else stem
            suffix = "backing_vocal" if stem == "backing_vocals" else suffix
            expected_events.extend([f"backend:{stem}", f"save:Alpha Song_{suffix}"])
        self.assertEqual(expected_events, events)
        self.assertIn("Alpha Song_woodwinds.wav", "\n".join(saved_paths))
        self.assertIn("Alpha Song_guitar.wav", "\n".join(saved_paths))
        self.assertIn("Alpha Song_vocal.wav", "\n".join(saved_paths))
        sample_sidecar = next(payload for payload in json_payloads if "_meta" in payload)
        request = sample_sidecar["_meta"]["request"]
        self.assertEqual("woodwinds", sample_sidecar["track_name"])
        self.assertEqual("woodwinds", sample_sidecar["extract_stem_name"])
        self.assertTrue(sample_sidecar["extract_all_stems"])
        self.assertEqual("", request["caption"])
        self.assertIsNone(request["track_name"])
        self.assertTrue(request["extract_all_stems"])
        self.assertEqual(TRACK_NAMES, request["extract_stem_names"])
        self.assertEqual(len(TRACK_NAMES), request["generation_count"])
        self.assertEqual(1, request["requested_generation_count"])
        self.assertIn("Alpha Song_vocal.json", "\n".join(final[16]))

    def test_extract_trim_shortens_saved_tensor_and_metadata(self):
        """Extract trim should modify saved audio tensors and sidecar metadata."""

        saved_shapes = []
        saved_peaks = []
        json_payloads = []
        trim_settings = []
        backend_params = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            backend_params.append(params)
            return _fake_tensor_result(str(config.seeds[0]))

        def fake_save_audio(audio_data, output_path, **_kwargs):
            saved_shapes.append(tuple(audio_data.shape))
            saved_peaks.append(float(audio_data.detach().abs().max().item()))
            return output_path

        def fake_write_json(_path, payload):
            json_payloads.append(payload)
            return str(_path)

        def fake_trim(audio_tensor, **kwargs):
            trim_settings.append(kwargs["trim_settings"])
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
                    track_name="vocals",
                    extract_trim_empty_output=True,
                    extract_trim_threshold_db=-42.0,
                    audio_processing_settings={
                        "trim_threshold_db": -35.0,
                        "trim_margin_seconds": 0.8,
                        "trim_mincut": 12,
                        "trim_minclip": 6,
                    },
                    save_audio_side_effect=fake_save_audio,
                    write_json_side_effect=fake_write_json,
                )

        self.assertEqual([(1, 4)], saved_shapes)
        self.assertFalse(backend_params[0].enable_normalization)
        self.assertAlmostEqual(0.8912509, saved_peaks[0], places=5)
        self.assertEqual([-35.0], [settings.threshold_db for settings in trim_settings])
        self.assertEqual([0.8], [settings.margin_seconds for settings in trim_settings])
        self.assertEqual([12], [settings.mincut for settings in trim_settings])
        self.assertEqual([6], [settings.minclip for settings in trim_settings])
        sample_sidecar = next(payload for payload in json_payloads if "extract_trim" in payload)
        self.assertTrue(sample_sidecar["extract_trim"]["applied"])
        self.assertEqual(2, sample_sidecar["extract_trim"]["segments"][0]["start_sample"])
        self.assertEqual(6, sample_sidecar["extract_trim"]["segments"][0]["end_sample"])

    def test_extract_all_stems_matches_single_stem_save_path_with_and_without_trim(self):
        """Extract-all-stems should save each stem like the matching single-stem run."""

        import torch

        def run_extract_case(*, extract_all_stems, trim_enabled):
            saved_tensors = {}
            backend_params = {}
            trim_settings = []

            def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
                _ = config, progress
                backend_params[params.caption] = params
                return _fake_tensor_result(params.caption)

            def fake_save_audio(audio_data, output_path, **_kwargs):
                saved_tensors[Path(output_path).stem] = audio_data.detach().clone()
                return output_path

            def fake_trim(audio_tensor, **kwargs):
                trim_settings.append(kwargs["trim_settings"])
                metadata = {
                    "enabled": True,
                    "applied": True,
                    "reason": "auto_editor_trimmed",
                    "segments": [{"start_sample": 2, "end_sample": 6}],
                }
                return SilenceTrimResult(audio_tensor[:, 2:6], metadata)

            with tempfile.TemporaryDirectory() as tmp, patch.object(
                generation_progress,
                "save_extract_remaining_audio",
                return_value={"applied": False},
            ), patch(
                "acestep.ui.gradio.events.results.generation_progress.trim_silent_edges",
                side_effect=fake_trim,
            ):
                self._run_generation(
                    tmp,
                    fake_generate_music,
                    task_type="extract",
                    src_audio=r"C:\music\Alpha Song.wav",
                    track_name=None if extract_all_stems else "vocals",
                    extract_all_stems=extract_all_stems,
                    audio_format="wav",
                    extract_trim_empty_output=trim_enabled,
                    audio_processing_settings={
                        "trim_threshold_db": -35.0,
                        "trim_margin_seconds": 0.8,
                        "trim_mincut": 12,
                        "trim_minclip": 6,
                    },
                    save_audio_side_effect=fake_save_audio,
                )

            return SimpleNamespace(
                saved_tensors=saved_tensors,
                backend_params=backend_params,
                trim_settings=trim_settings,
            )

        for trim_enabled in (False, True):
            with self.subTest(trim_enabled=trim_enabled):
                single = run_extract_case(extract_all_stems=False, trim_enabled=trim_enabled)
                all_stems = run_extract_case(extract_all_stems=True, trim_enabled=trim_enabled)

                torch.testing.assert_close(
                    single.saved_tensors["song-vocals"],
                    all_stems.saved_tensors["Alpha Song_vocal"],
                )
                self.assertEqual(
                    single.backend_params["vocals"].instruction,
                    all_stems.backend_params["vocals"].instruction,
                )
                self.assertEqual(
                    single.backend_params["vocals"].enable_normalization,
                    all_stems.backend_params["vocals"].enable_normalization,
                )
                expected_single_trim_calls = 1 if trim_enabled else 0
                expected_all_trim_calls = len(TRACK_NAMES) if trim_enabled else 0
                self.assertEqual(expected_single_trim_calls, len(single.trim_settings))
                self.assertEqual(expected_all_trim_calls, len(all_stems.trim_settings))
                if trim_enabled:
                    self.assertEqual(
                        [-35.0] * expected_all_trim_calls,
                        [settings.threshold_db for settings in all_stems.trim_settings],
                    )

    def test_extract_all_stems_request_metadata_ignores_per_stem_effective_instruction(self):
        """All-stems request metadata should not inherit the final stem instruction."""

        json_payloads = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            _ = config, progress
            result = _fake_tensor_result(params.caption)
            result.extra_outputs["effective_generation"] = {
                "requested_task_type": "extract",
                "task_type": "extract",
                "instruction": params.instruction,
                "caption": params.caption,
                "vocal_language": "unknown",
                "audio_duration": -1,
                "repainting_start": 0.0,
                "repainting_end": -1,
                "lyric_repaint_local_span": False,
            }
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            generation_progress,
            "save_extract_remaining_audio",
            return_value={"applied": False},
        ):
            self._run_generation(
                tmp,
                fake_generate_music,
                task_type="extract",
                src_audio=r"C:\music\Alpha Song.wav",
                track_name=None,
                extract_all_stems=True,
                audio_format="wav",
                write_json_side_effect=lambda _path, payload: json_payloads.append(payload),
            )

        sample_sidecar = next(payload for payload in json_payloads if "_meta" in payload)
        request = sample_sidecar["_meta"]["request"]
        generation_params = request["generation_params"]
        self.assertEqual("woodwinds", sample_sidecar["track_name"])
        self.assertEqual(TASK_INSTRUCTIONS["extract_default"], request["instruction"])
        self.assertEqual("", request["caption"])
        self.assertIsNone(request["track_name"])
        self.assertTrue(request["extract_all_stems"])
        self.assertEqual(TRACK_NAMES, request["extract_stem_names"])
        self.assertNotIn("effective_instruction", request)
        self.assertEqual(TASK_INSTRUCTIONS["extract_default"], generation_params["instruction"])
        self.assertNotIn("requested_instruction", generation_params)

    def test_lego_saves_raw_track_while_latest_area_uses_mix(self):
        """Lego latest-area preview should crop the mix and expose the raw layer separately."""

        import torch

        saved_paths = []
        latest_area_metadata = {
            "applied": True,
            "generated_area_path": "/tmp/song_latest_repainted_area.mp3",
            "original_area_path": "/tmp/song_latest_repainted_area_original.mp3",
        }

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            result = _fake_result(str(config.seeds[0]))
            result.extra_outputs["lego_layer_wavs"] = torch.ones(1, 2, 8) * 0.5
            return result

        def fake_save_audio(audio_data, output_path, **_kwargs):
            saved_paths.append((output_path, audio_data))
            return output_path

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            generation_progress,
            "create_latest_edit_area_clips",
            return_value=latest_area_metadata,
        ) as latest_area_mock:
            final = self._run_generation(
                tmp,
                fake_generate_music,
                task_type="lego",
                src_audio="source.wav",
                repainting_start=0.0,
                repainting_end=-1,
                save_audio_side_effect=fake_save_audio,
            )

        layer_paths = [
            path for path, _audio_data in saved_paths
            if path.endswith("_lego_generated_track.mp3")
        ]
        self.assertEqual(1, len(layer_paths))
        latest_area_mock.assert_called_once()
        self.assertNotEqual(
            latest_area_mock.call_args.kwargs["generated_audio_path"],
            layer_paths[0],
        )
        self.assertTrue(latest_area_mock.call_args.kwargs["generated_audio_path"].endswith(".mp3"))
        self.assertIn(layer_paths[0], final[16])
        self.assertIn(latest_area_metadata["generated_area_path"], final[16])
        self.assertIn(latest_area_metadata["original_area_path"], final[16])
        layer_audio = next(audio_data for path, audio_data in saved_paths if path == layer_paths[0])
        torch.testing.assert_close(layer_audio, torch.ones(2, 8) * 0.5)

    def test_lyric_repaint_request_payload_uses_effective_instruction(self):
        """Saved request metadata should show lyric repaint's text-to-song instruction."""

        json_payloads = []
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

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            result = _fake_result(str(config.seeds[0]))
            result.extra_outputs["effective_generation"] = effective_generation
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            generation_progress,
            "create_latest_edit_area_clips",
            return_value={"applied": False},
        ):
            self._run_generation(
                tmp,
                fake_generate_music,
                task_type="repaint",
                src_audio="source.wav",
                lyrics="new lyric line",
                instruction_display_gen=TASK_INSTRUCTIONS["repaint"],
                repainting_start=1.0,
                repainting_end=2.0,
                write_json_side_effect=lambda _path, payload: json_payloads.append(payload),
            )

        sample_sidecar = next(payload for payload in json_payloads if "_meta" in payload)
        request = sample_sidecar["_meta"]["request"]
        generation_params = request["generation_params"]
        self.assertEqual(DEFAULT_DIT_INSTRUCTION, request["instruction"])
        self.assertEqual("text2music", request["effective_task_type"])
        self.assertEqual(DEFAULT_DIT_INSTRUCTION, generation_params["instruction"])
        self.assertEqual(
            TASK_INSTRUCTIONS["repaint"],
            generation_params["requested_instruction"],
        )
        self.assertEqual("text2music", generation_params["effective_task_type"])
        self.assertTrue(generation_params["lyric_repaint_local_span"])

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
