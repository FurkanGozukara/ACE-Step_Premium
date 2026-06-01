"""Core audio generation with progressive UI yields.

Contains the main ``generate_with_progress`` generator that drives the
Gradio generate button: validates GPU limits, calls the inference
pipeline, saves audio files, and optionally runs auto-scoring and
auto-LRC in a single streaming pass.
"""
import os
import sys
import time as time_module

import gradio as gr
from loguru import logger

from acestep.audio_processing.generated_postprocess import postprocess_generated_sample
from acestep.audio_processing.silence_trim import trim_silent_edges
from acestep.audio_processing.settings import AudioProcessingSettings
from acestep.sam_audio_segment.generated_postprocess import (
    postprocess_generated_sample as postprocess_generated_sam_audio,
)
from acestep.sam_audio_segment.settings import SamAudioSettings
from acestep.core.generation.cancellation import check_generation_cancelled
from acestep.gpu_config import (
    get_global_gpu_config,
    check_duration_limit,
)
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.events.generation.generation_count import (
    normalize_generation_count,
    seed_for_generation_index,
)
from acestep.ui.gradio.events.generation_handlers import parse_and_validate_timesteps
from acestep.ui.gradio.events.generation.audio_format_options import (
    audio_file_extension,
    normalize_audio_format,
    output_audio_formats,
    primary_audio_format,
)
from acestep.ui.gradio.events.results.generation_info import (
    _build_generation_info,
)
from acestep.ui.gradio.events.results.generation_sequence import generate_sequential_songs
from acestep.ui.gradio.events.results.output_manager import (
    build_generation_manifest,
    create_generation_run_dir,
    persist_generation_inputs,
    write_json,
    write_text,
)
from acestep.ui.gradio.events.results.generation_task_type import resolve_no_fsq_task_type
from acestep.ui.gradio.events.results.audio_playback_updates import (
    build_audio_slot_update,
)
from acestep.ui.gradio.events.results.scoring import calculate_score_handler
from acestep.ui.gradio.events.results.lrc_utils import lrc_to_vtt_file
from acestep.ui.gradio.events.results.session_artifacts import persist_sample_session_artifacts


def _get_torch():
    """Return the already-imported torch module without importing it."""

    return sys.modules.get("torch")


def generate_with_progress(
    dit_handler, llm_handler,
    captions, lyrics, bpm, key_scale, time_signature, vocal_language,
    inference_steps, guidance_scale, random_seed_checkbox, seed,
    reference_audio, audio_duration, batch_size_input, src_audio,
    text2music_audio_code_string, repainting_start, repainting_end,
    instruction_display_gen, audio_cover_strength, cover_noise_strength, task_type,
    no_fsq, use_adg, cfg_interval_start, cfg_interval_end, shift, infer_method,
    sampler_mode, velocity_norm_threshold, velocity_ema_factor,
    dcw_enabled, dcw_mode, dcw_scaler, dcw_high_scaler, dcw_wavelet,
    custom_timesteps, audio_format, mp3_bitrate, mp3_sample_rate, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    auto_lrc,
    score_scale,
    lm_batch_chunk_size,
    enable_normalization,
    normalization_db,
    fade_in_duration,
    fade_out_duration,
    latent_shift,
    latent_rescale,
    repaint_mode,
    repaint_strength,
    retake_variance=0.0,
    retake_seed="",
    flow_edit_morph=False,
    flow_edit_source_caption="",
    flow_edit_source_lyrics="",
    flow_edit_n_min=0.0,
    flow_edit_n_max=1.0,
    flow_edit_n_avg=1,
    generate_lm_audio_codes=None,
    extract_trim_empty_output=False,
    extract_trim_threshold_db=-40.0,
    audio_processing_settings=None,
    sam_audio_settings=None,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate audio with progress tracking.

    This is a Gradio generator that yields partial UI updates as each
    sample is processed, enabling progressive display of results.

    Yields:
        Tuple of Gradio component updates for the 52-output generate event.
    """
    from acestep.audio_utils import save_audio
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    request_started_at = time_module.time()
    run_dir = create_generation_run_dir()
    logger.info(f"[generate_with_progress] Saving run outputs to {run_dir}")
    check_generation_cancelled()

    # GPU memory validation
    gpu_config = get_global_gpu_config()
    lm_initialized = llm_handler.llm_initialized if llm_handler else False

    # Save-memory mode: force-disable features that require intermediate tensors
    if gpu_config.save_memory_mode:
        auto_score = False
        auto_lrc = False

    if audio_duration is not None and audio_duration > 0:
        is_valid, warning_msg = check_duration_limit(audio_duration, gpu_config, lm_initialized)
        if not is_valid:
            gr.Warning(warning_msg)
            max_dur = gpu_config.max_duration_with_lm if lm_initialized else gpu_config.max_duration_without_lm
            audio_duration = min(audio_duration, max_dur)
            logger.warning(f"Duration clamped to {audio_duration}s due to GPU memory limits")

    generation_count = normalize_generation_count(batch_size_input)

    # Skip Phase 1 metas COT if sample is already formatted
    actual_use_cot_metas = use_cot_metas
    if is_format_caption and use_cot_metas:
        actual_use_cot_metas = False
        logger.info("[generate_with_progress] Skipping Phase 1 metas COT: is_format_caption=True")
        gr.Info(t("messages.skipping_metas_cot"))

    parsed_timesteps, _has_ts_warn, _ = parse_and_validate_timesteps(custom_timesteps, inference_steps)
    actual_inference_steps = len(parsed_timesteps) - 1 if parsed_timesteps is not None else inference_steps
    audio_format = normalize_audio_format(audio_format)
    backend_audio_format = primary_audio_format(audio_format)
    ap_settings = _audio_processing_settings(audio_processing_settings)
    sam_settings = _sam_audio_settings(sam_audio_settings)
    active_model = _active_dit_model_label(dit_handler)
    logger.info(
        f"[generate_with_progress] Generation request: model={active_model}, "
        f"inference_steps_requested={inference_steps}, "
        f"inference_steps_used={actual_inference_steps}, "
        f"songs={generation_count}, backend_batch_size=1, duration={audio_duration}"
    )

    task_type = resolve_no_fsq_task_type(task_type, bool(no_fsq))

    # text2music never uses src_audio EXCEPT when flow_edit_morph is on:
    # the morph overlay needs the source audio for ``zt_src``/``zt_tar``
    # formation in the V_delta integration.  Without this guard the UI
    # silently zeroed src_audio for Custom mode and the backend's morph
    # check then errored with "Flow-edit morph requires a source audio".
    if task_type == "text2music" and not flow_edit_morph:
        src_audio = None

    # Defensive guard: cover/repaint/extract/lego tasks should never use
    # stale audio codes from the text2music_audio_code_string textbox.
    # Only text2music (Custom mode) with thinking disabled should pass codes.
    if task_type != "text2music":
        text2music_audio_code_string = ""

    gen_params = GenerationParams(
        task_type=task_type,
        instruction=instruction_display_gen,
        reference_audio=reference_audio,
        src_audio=src_audio,
        audio_codes=text2music_audio_code_string if not think_checkbox else "",
        caption=captions or "",
        lyrics=lyrics or "",
        instrumental=False,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=key_scale,
        timesignature=time_signature,
        duration=audio_duration,
        inference_steps=actual_inference_steps,
        guidance_scale=guidance_scale,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        shift=shift,
        infer_method=infer_method,
        sampler_mode=sampler_mode,
        velocity_norm_threshold=velocity_norm_threshold,
        velocity_ema_factor=velocity_ema_factor,
        dcw_enabled=dcw_enabled,
        dcw_mode=dcw_mode,
        dcw_scaler=dcw_scaler,
        dcw_high_scaler=dcw_high_scaler,
        dcw_wavelet=dcw_wavelet,
        timesteps=parsed_timesteps,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        audio_cover_strength=audio_cover_strength,
        cover_noise_strength=cover_noise_strength,
        thinking=think_checkbox,
        generate_lm_audio_codes=generate_lm_audio_codes,
        lm_temperature=lm_temperature,
        lm_cfg_scale=lm_cfg_scale,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_negative_prompt=lm_negative_prompt,
        use_cot_metas=actual_use_cot_metas,
        use_cot_caption=use_cot_caption,
        use_cot_language=use_cot_language,
        use_constrained_decoding=True,
        enable_normalization=enable_normalization,
        normalization_db=normalization_db,
        fade_in_duration=fade_in_duration if fade_in_duration else 0.0,
        fade_out_duration=fade_out_duration if fade_out_duration else 0.0,
        latent_shift=latent_shift,
        latent_rescale=latent_rescale,
        repaint_mode=repaint_mode if repaint_mode else "balanced",
        repaint_strength=float(repaint_strength) if repaint_strength is not None else 0.5,
        retake_variance=float(retake_variance) if retake_variance is not None else 0.0,
        # Empty textbox -> None; otherwise a string is fine (handler.prepare_seeds parses it).
        retake_seed=(retake_seed.strip() or None) if isinstance(retake_seed, str) else retake_seed,
        flow_edit_morph=bool(flow_edit_morph),
        flow_edit_source_caption=flow_edit_source_caption or "",
        flow_edit_source_lyrics=flow_edit_source_lyrics or "",
        flow_edit_n_min=float(flow_edit_n_min) if flow_edit_n_min is not None else 0.0,
        flow_edit_n_max=float(flow_edit_n_max) if flow_edit_n_max is not None else 1.0,
        flow_edit_n_avg=int(flow_edit_n_avg) if flow_edit_n_avg is not None else 1,
    )

    gen_config = GenerationConfig(
        batch_size=1,
        allow_lm_batch=False,
        use_random_seed=random_seed_checkbox,
        seeds=seed_for_generation_index(
            seed,
            0,
            random_seed=bool(random_seed_checkbox),
        ),
        lm_batch_chunk_size=lm_batch_chunk_size,
        constrained_decoding_debug=constrained_decoding_debug,
        audio_format=backend_audio_format,
        mp3_bitrate=mp3_bitrate,
        mp3_sample_rate=mp3_sample_rate,
    )

    result = generate_sequential_songs(
        generate_music,
        dit_handler,
        llm_handler,
        params=gen_params,
        base_config=gen_config,
        generation_count=generation_count,
        seed=seed,
        random_seed=bool(random_seed_checkbox),
        progress=progress,
    )
    check_generation_cancelled()

    all_audio_paths: list = []

    seed_value_for_ui = result.extra_outputs.get("seed_value", "")
    lm_generated_metadata = result.extra_outputs.get("lm_metadata", {})
    time_costs = result.extra_outputs.get("time_costs", {}).copy()

    audio_conversion_start_time = time_module.time()
    total_auto_score_time = 0.0
    total_auto_lrc_time = 0.0

    sample_manifest_rows: list[dict[str, object]] = []
    generation_info = _build_generation_info(
        lm_metadata=lm_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios) if result.success else 0,
        audio_format=audio_format,
    )
    request_payload = _build_request_payload(
        captions=captions,
        lyrics=lyrics,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        inference_steps=inference_steps,
        actual_inference_steps=actual_inference_steps,
        guidance_scale=guidance_scale,
        random_seed_checkbox=random_seed_checkbox,
        seed=seed,
        seed_value_for_ui=seed_value_for_ui,
        reference_audio=reference_audio,
        audio_duration=audio_duration,
        batch_size_input=generation_count,
        src_audio=src_audio,
        text2music_audio_code_string=text2music_audio_code_string,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        instruction_display_gen=instruction_display_gen,
        audio_cover_strength=audio_cover_strength,
        cover_noise_strength=cover_noise_strength,
        task_type=task_type,
        no_fsq=no_fsq,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        shift=shift,
        infer_method=infer_method,
        sampler_mode=sampler_mode,
        velocity_norm_threshold=velocity_norm_threshold,
        velocity_ema_factor=velocity_ema_factor,
        dcw_enabled=dcw_enabled,
        dcw_mode=dcw_mode,
        dcw_scaler=dcw_scaler,
        dcw_high_scaler=dcw_high_scaler,
        dcw_wavelet=dcw_wavelet,
        custom_timesteps=custom_timesteps,
        parsed_timesteps=parsed_timesteps,
        audio_format=audio_format,
        mp3_bitrate=mp3_bitrate,
        mp3_sample_rate=mp3_sample_rate,
        lm_temperature=lm_temperature,
        think_checkbox=think_checkbox,
        generate_lm_audio_codes=generate_lm_audio_codes,
        lm_cfg_scale=lm_cfg_scale,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_negative_prompt=lm_negative_prompt,
        use_cot_metas=use_cot_metas,
        use_cot_caption=use_cot_caption,
        use_cot_language=use_cot_language,
        is_format_caption=is_format_caption,
        constrained_decoding_debug=constrained_decoding_debug,
        allow_lm_batch=allow_lm_batch,
        auto_score=auto_score,
        auto_lrc=auto_lrc,
        score_scale=score_scale,
        lm_batch_chunk_size=lm_batch_chunk_size,
        enable_normalization=enable_normalization,
        normalization_db=normalization_db,
        fade_in_duration=fade_in_duration,
        fade_out_duration=fade_out_duration,
        latent_shift=latent_shift,
        latent_rescale=latent_rescale,
        repaint_mode=repaint_mode,
        repaint_strength=repaint_strength,
        retake_variance=retake_variance,
        retake_seed=retake_seed,
        flow_edit_morph=flow_edit_morph,
        flow_edit_source_caption=flow_edit_source_caption,
        flow_edit_source_lyrics=flow_edit_source_lyrics,
        flow_edit_n_min=flow_edit_n_min,
        flow_edit_n_max=flow_edit_n_max,
        flow_edit_n_avg=flow_edit_n_avg,
        extract_trim_empty_output=extract_trim_empty_output,
        extract_trim_threshold_db=extract_trim_threshold_db,
        audio_processing_settings=ap_settings.to_payload(),
        sam_audio_settings=sam_settings.to_payload(),
        gen_params=gen_params,
        gen_config=gen_config,
        gpu_config=gpu_config,
        lm_initialized=lm_initialized,
        lm_generated_metadata=lm_generated_metadata,
        service_metadata=_build_service_metadata(dit_handler, llm_handler),
    )
    run_assets = persist_generation_inputs(
        run_dir=run_dir,
        caption=captions,
        lyrics=lyrics,
        reference_audio=reference_audio,
        src_audio=src_audio,
        request_payload=request_payload,
    )
    request_payload["saved_run_assets"] = run_assets
    run_asset_paths = [
        path
        for key, path in run_assets.items()
        if key.endswith("_path") and path
    ]

    if not result.success:
        build_generation_manifest(
            run_dir=run_dir,
            request_started_at=request_started_at,
            request_finished_at=time_module.time(),
            generation_info=generation_info,
            seed_value=seed_value_for_ui,
            audio_format=audio_format,
            sample_files=[],
            time_costs=time_costs,
            request_payload=request_payload,
            lm_metadata=lm_generated_metadata,
            status="failed",
        )
        yield (
            (None,) * 8
            + (None, generation_info, result.status_message, gr.skip())
            + (gr.skip(),) * 8  # scores
            + (gr.skip(),) * 8  # codes_display
            + (gr.skip(),) * 8  # details_accordion
            + (gr.skip(),) * 8  # lrc_display
            + (None, is_format_caption, None, None)
        )
        return

    audios = result.audios
    visible_slots = 8
    stored_sample_count = max(visible_slots, len(audios))
    audio_outputs = [None] * visible_slots
    final_codes_list = [""] * stored_sample_count
    final_scores_list = [""] * visible_slots
    final_lrcs_list = [""] * stored_sample_count
    final_lrc_paths_list = [None] * stored_sample_count
    final_subtitles_list = [None] * stored_sample_count
    progress(0.99, "Preparing audio files...")

    # Clear all scores/codes/lrc displays
    clear_scores = [gr.update(value="", visible=True) for _ in range(8)]
    clear_codes = [gr.update(value="", visible=True) for _ in range(8)]
    clear_lrcs = [gr.update(value="", visible=True) for _ in range(8)]
    clear_accordions = [gr.skip() for _ in range(8)]
    # Keep existing players mounted during generation to avoid browser volume reset.
    dump_audio = [gr.skip()] * 8

    yield (
        *dump_audio,
        None, generation_info,
        f"Preparing generation... Model: {active_model}; steps: {actual_inference_steps}",
        gr.skip(),
        *clear_scores, *clear_codes, *clear_accordions, *clear_lrcs,
        lm_generated_metadata, is_format_caption, None, None,
    )
    time_module.sleep(0.1)

    for i, dit_audio in enumerate(audios):
        check_generation_cancelled()
        key = dit_audio["key"]
        audio_tensor = dit_audio["tensor"]
        sample_rate = dit_audio["sample_rate"]
        audio_params = dit_audio["params"]
        audio_tensor, extract_trim_metadata = _trim_extract_audio(
            audio_tensor,
            sample_rate=sample_rate,
            task_type=task_type,
            enabled=extract_trim_empty_output,
            threshold_db=extract_trim_threshold_db,
        )
        audio_params["extract_trim"] = extract_trim_metadata

        temp_dir = str(run_dir.resolve()).replace("\\", "/")
        json_path = os.path.join(temp_dir, f"{key}.json").replace("\\", "/")

        saved_audio_paths: dict[str, str] = {}
        for concrete_format in output_audio_formats(audio_format):
            ext = audio_file_extension(concrete_format)
            target_path = os.path.join(temp_dir, f"{key}.{ext}").replace("\\", "/")
            saved_path = save_audio(
                audio_data=audio_tensor,
                output_path=target_path,
                sample_rate=sample_rate,
                format=concrete_format,
                channels_first=True,
                mp3_bitrate=mp3_bitrate,
                mp3_sample_rate=mp3_sample_rate,
            )
            if saved_path:
                saved_audio_paths[concrete_format] = saved_path.replace("\\", "/")

        audio_path = saved_audio_paths.get(backend_audio_format)
        if not audio_path and saved_audio_paths:
            audio_path = next(iter(saved_audio_paths.values()))
        audio_path = audio_path or ""
        original_audio_paths = dict(saved_audio_paths)
        postprocess_metadata = postprocess_generated_sample(
            source_audio_path=audio_path,
            run_dir=temp_dir,
            key=key,
            settings=ap_settings,
            original_audio_paths=original_audio_paths,
        )
        if postprocess_metadata.get("applied"):
            processed_path = str(postprocess_metadata.get("audio_path") or "")
            if processed_path:
                saved_audio_paths = _postprocessed_audio_paths(
                    original_audio_paths,
                    processed_path,
                    ap_settings,
                )
                audio_path = processed_path
        audio_params["audio_format"] = audio_format
        audio_params["primary_audio_format"] = backend_audio_format
        audio_params["saved_audio_formats"] = list(saved_audio_paths.keys())
        audio_params["audio_paths"] = saved_audio_paths
        audio_params["audio_processing"] = postprocess_metadata
        if "mp3" in saved_audio_paths:
            audio_params["mp3_path"] = saved_audio_paths["mp3"]

        sam_postprocess_metadata = postprocess_generated_sam_audio(
            source_audio_path=audio_path,
            run_dir=temp_dir,
            key=key,
            settings=sam_settings,
        )
        if sam_postprocess_metadata.get("applied"):
            sam_target_path = str(sam_postprocess_metadata.get("target_audio_path") or "")
            if sam_target_path:
                saved_audio_paths = _sam_audio_paths(
                    saved_audio_paths,
                    sam_target_path,
                    sam_settings,
                )
                audio_path = sam_target_path
                if i < visible_slots:
                    audio_outputs[i] = audio_path
        audio_params["sam_audio"] = sam_postprocess_metadata
        audio_params["saved_audio_formats"] = list(saved_audio_paths.keys())
        audio_params["audio_paths"] = saved_audio_paths

        _persist_repaint_source_latents(
            source_latents=_extract_repaint_source_latents(result.extra_outputs, i),
            json_path=json_path,
            audio_params=audio_params,
        )
        persist_sample_session_artifacts(
            extra_outputs=result.extra_outputs,
            sample_idx=i,
            json_path=json_path,
            audio_params=audio_params,
        )

        if i < visible_slots:
            audio_outputs[i] = audio_path
        all_audio_paths.extend(path for path in saved_audio_paths.values() if path)
        if postprocess_metadata.get("metadata_path"):
            all_audio_paths.append(str(postprocess_metadata["metadata_path"]))
        all_audio_paths.extend(str(path) for path in sam_postprocess_metadata.get("files", []) if path)
        all_audio_paths.append(json_path)

        code_str = audio_params.get("audio_codes", "")
        final_codes_list[i] = code_str

        scores_ui_updates = [gr.skip() for _ in range(8)]
        score_str = "Done!"

        if auto_score:
            auto_score_start = time_module.time()
            sample_tensor_data = _extract_sample_tensor(result.extra_outputs, i)
            score_str = calculate_score_handler(
                llm_handler, code_str, captions, lyrics, lm_generated_metadata,
                bpm, key_scale, time_signature, audio_duration, vocal_language,
                score_scale, dit_handler, sample_tensor_data, inference_steps,
            )
            total_auto_score_time += time_module.time() - auto_score_start

        if i < visible_slots:
            scores_ui_updates[i] = score_str
            final_scores_list[i] = score_str

        if auto_lrc:
            auto_lrc_start = time_module.time()
            _run_auto_lrc(
                dit_handler, result.extra_outputs, i,
                audio_duration, vocal_language, inference_steps, json_path,
                final_lrcs_list, final_lrc_paths_list, final_subtitles_list,
            )
            total_auto_lrc_time += time_module.time() - auto_lrc_start

        sample_row = {
            "sample_index": i + 1,
            "key": key,
            "audio_path": audio_path,
            "audio_paths": saved_audio_paths,
            "mp3_path": saved_audio_paths.get("mp3"),
            "audio_processing": postprocess_metadata,
            "sam_audio": sam_postprocess_metadata,
            "metadata_path": json_path,
            "audio_format": audio_format,
            "primary_audio_format": backend_audio_format,
            "saved_audio_formats": list(saved_audio_paths.keys()),
            "sample_rate": sample_rate,
            "score": score_str,
            "audio_codes": code_str,
            "lrc": final_lrcs_list[i],
            "lrc_path": final_lrc_paths_list[i],
            "subtitle_path": final_subtitles_list[i],
            "params": audio_params,
        }
        sample_manifest_rows.append(sample_row)
        write_json(
            json_path,
            {
                **audio_params,
                "_meta": {
                    "sample_index": i + 1,
                    "run_dir": temp_dir,
                    "audio_path": audio_path,
                    "audio_paths": saved_audio_paths,
                    "audio_format": audio_format,
                    "primary_audio_format": backend_audio_format,
                    "saved_audio_formats": list(saved_audio_paths.keys()),
                    "sample_rate": sample_rate,
                    "score": score_str,
                    "lrc_path": final_lrc_paths_list[i],
                    "subtitle_path": final_subtitles_list[i],
                    "generation_info": generation_info,
                    "lm_metadata": lm_generated_metadata,
                    "request": request_payload,
                },
                "lrc": final_lrcs_list[i],
                "lrc_path": final_lrc_paths_list[i],
            },
        )

        # STEP 1: yield audio + clear LRC
        cur_audio = [gr.skip()] * visible_slots
        cur_codes = [gr.skip()] * visible_slots
        cur_accordions = [gr.skip()] * 8
        lrc_clear = [gr.skip()] * visible_slots
        if i < visible_slots:
            cur_audio[i] = build_audio_slot_update(gr, audio_path)
            cur_codes[i] = gr.update(value=code_str, visible=True)
            lrc_clear[i] = gr.update(value="", visible=True)

        yield (
            *cur_audio,
            all_audio_paths, generation_info, f"Encoding & Ready: {i + 1}/{len(audios)}", seed_value_for_ui,
            *scores_ui_updates, *cur_codes, *cur_accordions, *lrc_clear,
            lm_generated_metadata, is_format_caption, None, None,
        )
        time_module.sleep(0.05)

        # STEP 2: set actual LRC (triggers .change() for subtitles)
        if i < visible_slots and final_lrcs_list[i]:
            skip8 = [gr.skip()] * visible_slots
            lrc_set = [gr.skip()] * visible_slots
            lrc_set[i] = gr.update(value=final_lrcs_list[i], visible=True)
            yield (
                *skip8,
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                *skip8, *skip8, *skip8, *lrc_set,
                gr.skip(), gr.skip(), None, None,
            )

        time_module.sleep(0.05)

    # Final timing
    check_generation_cancelled()
    audio_conversion_time = time_module.time() - audio_conversion_start_time
    if audio_conversion_time > 0:
        time_costs['audio_conversion_time'] = audio_conversion_time
    if total_auto_score_time > 0:
        time_costs['auto_score_time'] = total_auto_score_time
    if total_auto_lrc_time > 0:
        time_costs['auto_lrc_time'] = total_auto_lrc_time
    if 'pipeline_total_time' in time_costs:
        time_costs['pipeline_total_time'] += audio_conversion_time + total_auto_score_time + total_auto_lrc_time

    generation_info = _build_generation_info(
        lm_metadata=lm_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios),
        audio_format=audio_format,
    )
    manifest_path = build_generation_manifest(
        run_dir=run_dir,
        request_started_at=request_started_at,
        request_finished_at=time_module.time(),
        generation_info=generation_info,
        seed_value=seed_value_for_ui,
        audio_format=audio_format,
        sample_files=sample_manifest_rows,
        time_costs=time_costs,
        request_payload={
            **request_payload,
        },
        lm_metadata=lm_generated_metadata,
        status="completed",
    )
    all_audio_paths.append(manifest_path)
    all_audio_paths.extend(run_asset_paths)

    audio_playback_updates = []
    for idx in range(8):
        path = audio_outputs[idx]
        if path:
            audio_playback_updates.append(build_audio_slot_update(gr, path))
            logger.info(f"[generate_with_progress] Audio {idx + 1} path: {path}")
        else:
            audio_playback_updates.append(build_audio_slot_update(gr, None))

    final_codes_display = [gr.skip()] * 8
    final_accordions = [gr.skip()] * 8

    extra_to_store = _strip_extra_output_tensors(
        {
            **result.extra_outputs,
            "lrcs": final_lrcs_list,
            "lrc_paths": final_lrc_paths_list,
            "subtitles": final_subtitles_list,
        }
    )

    yield (
        *audio_playback_updates,
        all_audio_paths, generation_info, "Generation Complete", seed_value_for_ui,
        *final_scores_list, *final_codes_display, *final_accordions,
        *final_lrcs_list[:visible_slots],
        lm_generated_metadata, is_format_caption,
        extra_to_store,
        final_codes_list,
    )


def _extract_sample_tensor(extra_outputs, sample_idx):
    """Slice per-sample tensor data from *extra_outputs* for scoring.

    Returns ``None`` when data is missing or incomplete.
    """
    try:
        full_pred = extra_outputs.get("pred_latents")
        if full_pred is None or sample_idx >= full_pred.shape[0]:
            return None
        data = {
            "pred_latent": full_pred[sample_idx:sample_idx + 1],
            "encoder_hidden_states": extra_outputs.get("encoder_hidden_states")[sample_idx:sample_idx + 1]
                if extra_outputs.get("encoder_hidden_states") is not None else None,
            "encoder_attention_mask": extra_outputs.get("encoder_attention_mask")[sample_idx:sample_idx + 1]
                if extra_outputs.get("encoder_attention_mask") is not None else None,
            "context_latents": extra_outputs.get("context_latents")[sample_idx:sample_idx + 1]
                if extra_outputs.get("context_latents") is not None else None,
            "lyric_token_ids": extra_outputs.get("lyric_token_idss")[sample_idx:sample_idx + 1]
                if extra_outputs.get("lyric_token_idss") is not None else None,
        }
        if any(v is None for v in data.values()):
            return None
        return data
    except Exception as e:
        logger.warning(
            "[Auto Score] Failed to prepare tensor data for sample {}: {}", sample_idx, e
        )
        return None


def _build_request_payload(
    *,
    captions,
    lyrics,
    bpm,
    key_scale,
    time_signature,
    vocal_language,
    inference_steps,
    actual_inference_steps,
    guidance_scale,
    random_seed_checkbox,
    seed,
    seed_value_for_ui,
    reference_audio,
    audio_duration,
    batch_size_input,
    src_audio,
    text2music_audio_code_string,
    repainting_start,
    repainting_end,
    instruction_display_gen,
    audio_cover_strength,
    cover_noise_strength,
    task_type,
    no_fsq,
    use_adg,
    cfg_interval_start,
    cfg_interval_end,
    shift,
    infer_method,
    sampler_mode,
    velocity_norm_threshold,
    velocity_ema_factor,
    dcw_enabled,
    dcw_mode,
    dcw_scaler,
    dcw_high_scaler,
    dcw_wavelet,
    custom_timesteps,
    parsed_timesteps,
    audio_format,
    mp3_bitrate,
    mp3_sample_rate,
    lm_temperature,
    think_checkbox,
    generate_lm_audio_codes,
    lm_cfg_scale,
    lm_top_k,
    lm_top_p,
    lm_negative_prompt,
    use_cot_metas,
    use_cot_caption,
    use_cot_language,
    is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    auto_lrc,
    score_scale,
    lm_batch_chunk_size,
    enable_normalization,
    normalization_db,
    fade_in_duration,
    fade_out_duration,
    latent_shift,
    latent_rescale,
    repaint_mode,
    repaint_strength,
    retake_variance,
    retake_seed,
    flow_edit_morph,
    flow_edit_source_caption,
    flow_edit_source_lyrics,
    flow_edit_n_min,
    flow_edit_n_max,
    flow_edit_n_avg,
    extract_trim_empty_output,
    extract_trim_threshold_db,
    audio_processing_settings,
    sam_audio_settings,
    gen_params,
    gen_config,
    gpu_config,
    lm_initialized,
    lm_generated_metadata,
    service_metadata,
):
    """Build the full persisted request payload for a generation run."""
    return {
        "active_config_path": (
            service_metadata.get("dit_last_init_params", {}) or {}
        ).get("config_path"),
        "caption": captions or "",
        "lyrics": lyrics or "",
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "audio_duration": audio_duration,
        "generation_count": batch_size_input,
        "batch_size": 1,
        "task_type": task_type,
        "no_fsq": no_fsq,
        "instruction": instruction_display_gen,
        "guidance_scale": guidance_scale,
        "inference_steps_requested": inference_steps,
        "inference_steps_used": actual_inference_steps,
        "parsed_timesteps": parsed_timesteps,
        "custom_timesteps": custom_timesteps,
        "thinking": think_checkbox,
        "generate_lm_audio_codes": generate_lm_audio_codes,
        "audio_format": audio_format,
        "saved_audio_formats": output_audio_formats(audio_format),
        "primary_audio_format": primary_audio_format(audio_format),
        "mp3_bitrate": mp3_bitrate,
        "mp3_sample_rate": mp3_sample_rate,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "text2music_audio_code_string": text2music_audio_code_string,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "audio_cover_strength": audio_cover_strength,
        "cover_noise_strength": cover_noise_strength,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "infer_method": infer_method,
        "sampler_mode": sampler_mode,
        "velocity_norm_threshold": velocity_norm_threshold,
        "velocity_ema_factor": velocity_ema_factor,
        "dcw_enabled": dcw_enabled,
        "dcw_mode": dcw_mode,
        "dcw_scaler": dcw_scaler,
        "dcw_high_scaler": dcw_high_scaler,
        "dcw_wavelet": dcw_wavelet,
        "random_seed_checkbox": random_seed_checkbox,
        "seed_input": seed,
        "resolved_seed_value": seed_value_for_ui,
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "is_format_caption": is_format_caption,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "auto_lrc": auto_lrc,
        "score_scale": score_scale,
        "lm_batch_chunk_size": lm_batch_chunk_size,
        "enable_normalization": enable_normalization,
        "normalization_db": normalization_db,
        "fade_in_duration": fade_in_duration,
        "fade_out_duration": fade_out_duration,
        "latent_shift": latent_shift,
        "latent_rescale": latent_rescale,
        "repaint_mode": repaint_mode,
        "repaint_strength": repaint_strength,
        "retake_variance": retake_variance,
        "retake_seed": retake_seed,
        "flow_edit_morph": flow_edit_morph,
        "flow_edit_source_caption": flow_edit_source_caption,
        "flow_edit_source_lyrics": flow_edit_source_lyrics,
        "flow_edit_n_min": flow_edit_n_min,
        "flow_edit_n_max": flow_edit_n_max,
        "flow_edit_n_avg": flow_edit_n_avg,
        "extract_trim_empty_output": extract_trim_empty_output,
        "extract_trim_threshold_db": extract_trim_threshold_db,
        "audio_processing_settings": audio_processing_settings,
        "sam_audio_settings": sam_audio_settings,
        "generation_params": vars(gen_params),
        "generation_config": {
            **vars(gen_config),
            "generation_count": batch_size_input,
        },
        "runtime": {
            "lm_initialized_at_start": lm_initialized,
            "save_memory_mode": gpu_config.save_memory_mode,
            "lm_generated_metadata": lm_generated_metadata,
            **service_metadata,
        },
    }


def _build_service_metadata(dit_handler, llm_handler):
    """Return serializable service/runtime metadata for generated songs."""

    dit_model = getattr(dit_handler, "model", None)
    dit_config = getattr(dit_model, "config", None) or getattr(dit_handler, "config", None)
    return {
        "dit_model_class": type(dit_model).__name__ if dit_model is not None else None,
        "dit_model_module": type(dit_model).__module__ if dit_model is not None else None,
        "dit_config_model_version": getattr(dit_config, "model_version", None),
        "dit_config_is_turbo": getattr(dit_config, "is_turbo", None),
        "dit_quantization": getattr(dit_handler, "quantization", None),
        "dit_device": str(getattr(dit_handler, "device", "")),
        "dit_dtype": str(getattr(dit_handler, "dtype", "")),
        "dit_last_init_params": getattr(dit_handler, "last_init_params", {}) or {},
        "llm_last_init_params": getattr(llm_handler, "last_init_params", {}) or {},
    }


def _active_dit_model_label(dit_handler):
    """Return the current DiT model label for logs and transient status."""

    last_init_params = getattr(dit_handler, "last_init_params", {}) or {}
    config_path = str(last_init_params.get("config_path") or "").strip()
    return config_path or "unknown"


def _audio_processing_settings(raw_settings):
    """Return normalized audio-processing settings from a payload or object."""

    if isinstance(raw_settings, AudioProcessingSettings):
        return raw_settings
    return AudioProcessingSettings.from_payload(raw_settings)


def _sam_audio_settings(raw_settings):
    """Return normalized SAM-Audio settings from a payload or object."""

    if isinstance(raw_settings, SamAudioSettings):
        return raw_settings
    return SamAudioSettings.from_payload(raw_settings)


def _trim_extract_audio(audio_tensor, *, sample_rate, task_type, enabled, threshold_db):
    """Return trimmed ACE-Step Extract audio plus metadata."""

    normalized_task_type = str(task_type or "").strip()
    should_trim = bool(enabled) and normalized_task_type == "extract"
    if not should_trim:
        return audio_tensor, {
            "enabled": bool(enabled),
            "applied": False,
            "reason": "disabled" if not enabled else "non_extract_task",
            "mode": "auto_editor",
            "task_type": normalized_task_type,
        }

    trim_result = trim_silent_edges(
        audio_tensor,
        sample_rate=int(sample_rate or 48000),
        enabled=True,
    )
    metadata = dict(trim_result.metadata)
    metadata["task_type"] = normalized_task_type
    return trim_result.audio, metadata


def _postprocessed_audio_paths(original_paths, processed_path, settings):
    """Return visible audio paths after generated-song post-processing."""

    processed_key = f"postprocessed_{settings.output_format}"
    if settings.preserve_original:
        return {**original_paths, processed_key: processed_path}
    return {processed_key: processed_path}


def _sam_audio_paths(current_paths, target_path, settings):
    """Return visible audio paths after generated-song SAM-Audio processing."""

    processed_key = f"sam_audio_{settings.output_format}"
    if settings.preserve_original:
        return {**current_paths, processed_key: target_path}
    return {processed_key: target_path}


def _extract_repaint_source_latents(extra_outputs, sample_idx):
    """Return final generated latents for repaint-source reuse."""
    try:
        pred_latents = extra_outputs.get("pred_latents")
        if pred_latents is None or sample_idx >= pred_latents.shape[0]:
            return None
        return pred_latents[sample_idx]
    except (AttributeError, IndexError, TypeError):
        return None


def _strip_extra_output_tensors(extra_outputs):
    """Return extra outputs without tensor values for batch-queue storage."""
    torch = _get_torch()
    if torch is None:
        return extra_outputs
    return {
        key: value
        for key, value in extra_outputs.items()
        if not isinstance(value, torch.Tensor)
    }


def _persist_repaint_source_latents(source_latents, json_path: str, audio_params: dict) -> None:
    """Persist repaint-ready source latents beside a generated audio sidecar.

    The cached tensor is the final generated latent returned by the DiT path.
    This avoids a lossy decode-to-audio then VAE-reencode cycle for generated
    sources while uploaded audio keeps the normal repaint path.
    """
    if source_latents is None:
        return
    latent_path = os.path.splitext(json_path)[0] + ".repaint_latents.npy"
    try:
        import numpy as np

        np.save(latent_path, source_latents.detach().cpu().float().numpy())
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("[repaint_cache] Could not persist repaint source latents: {}", exc)
        return
    audio_params["repaint_source_latents_file"] = os.path.basename(latent_path)


def _run_auto_lrc(dit_handler, extra_outputs, sample_idx,
                  audio_duration, vocal_language, inference_steps, json_path,
                  final_lrcs_list, final_lrc_paths_list, final_subtitles_list):
    """Run automatic LRC generation for a single sample in-place.

    Updates *final_lrcs_list* and *final_subtitles_list* at *sample_idx*.
    """
    logger.info(f"[auto_lrc] Starting LRC generation for sample {sample_idx + 1}")
    try:
        pred_latents = extra_outputs.get("pred_latents")
        enc_hs = extra_outputs.get("encoder_hidden_states")
        enc_am = extra_outputs.get("encoder_attention_mask")
        ctx_lat = extra_outputs.get("context_latents")
        lyric_ids = extra_outputs.get("lyric_token_idss")

        required_tensors = [pred_latents, enc_hs, enc_am, ctx_lat, lyric_ids]
        if not all(x is not None for x in required_tensors):
            logger.warning(f"[auto_lrc] Missing required extra_outputs for sample {sample_idx + 1}")
            return

        actual_duration = audio_duration
        if actual_duration is None or actual_duration <= 0:
            actual_duration = pred_latents.shape[1] / 25.0

        lrc_result = dit_handler.get_lyric_timestamp(
            pred_latent=pred_latents[sample_idx:sample_idx + 1],
            encoder_hidden_states=enc_hs[sample_idx:sample_idx + 1],
            encoder_attention_mask=enc_am[sample_idx:sample_idx + 1],
            context_latents=ctx_lat[sample_idx:sample_idx + 1],
            lyric_token_ids=lyric_ids[sample_idx:sample_idx + 1],
            total_duration_seconds=float(actual_duration),
            vocal_language=vocal_language or "en",
            inference_steps=int(inference_steps),
            seed=42,
        )

        if lrc_result.get("success"):
            lrc_text = lrc_result.get("lrc_text", "")
            if not lrc_text:
                return
            final_lrcs_list[sample_idx] = lrc_text
            logger.info(f"[auto_lrc] LRC text length for sample {sample_idx + 1}: {len(lrc_text)}")
            lrc_path = os.path.splitext(json_path)[0] + ".lrc"
            vtt_target_path = os.path.splitext(json_path)[0] + ".vtt"
            write_text(lrc_path, lrc_text)
            vtt_path = lrc_to_vtt_file(
                lrc_text,
                total_duration=float(actual_duration),
                output_path=vtt_target_path,
            )
            final_lrc_paths_list[sample_idx] = lrc_path.replace("\\", "/")
            final_subtitles_list[sample_idx] = vtt_path
    except Exception as e:
        logger.warning(f"[auto_lrc] Failed to generate LRC for sample {sample_idx + 1}: {e}")
