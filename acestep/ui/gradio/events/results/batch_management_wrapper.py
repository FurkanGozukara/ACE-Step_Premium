"""Foreground batch generation wrapper for UI streaming updates."""

import gc
import inspect
import os
import sys
import time as time_module
from pathlib import Path

import gradio as gr
from loguru import logger

from acestep.gpu_config import find_best_lm_model_on_disk, get_global_gpu_config
from acestep.core.generation.cancellation import (
    CANCEL_MESSAGE,
    GenerationCancelled,
    check_generation_cancelled,
    cleanup_runtime_memory,
    generation_cancel_scope,
    is_generation_cancelled,
)
from acestep.model_downloader import DEFAULT_TURBO_DIT_MODEL
from acestep.ui.gradio.events.results.batch_management_helpers import (
    _apply_param_defaults,
    _build_saved_params,
    _extract_scores,
    _extract_ui_core_outputs,
    apply_lora_selection_for_generation,
    resolve_effective_lora_path,
)
from acestep.ui.gradio.events.results.batch_queue import (
    store_batch_in_queue,
    update_batch_indicator,
    update_navigation_buttons,
)
from acestep.ui.gradio.events.results.generation_info import IS_WINDOWS
from acestep.ui.gradio.events.results.generation_progress import generate_with_progress
from acestep.ui.gradio.events.results.subprocess_generation import (
    build_final_core_outputs,
    build_pending_core_outputs,
    stream_subprocess_generation,
)
from acestep.ui.gradio.events.results.output_manager import (
    get_active_generation_run_name,
    get_active_results_dir,
)
from acestep.ui.gradio.events.generation.generation_count import normalize_generation_count
from acestep.ui.gradio.events.generation.quantization import select_quantization_value
from acestep.ui.gradio.i18n import t

torch = sys.modules.get("torch")


def _get_torch():
    """Return the already-imported torch module without importing it."""

    global torch
    if torch is not None:
        return torch
    torch = sys.modules.get("torch")
    return torch


def _get_torch_for_in_process():
    """Import torch only for the in-process generation path."""

    global torch
    torch = _get_torch()
    if torch is not None:
        return torch
    import torch as imported_torch

    torch = imported_torch
    return imported_torch


def _select_quantization_value(
    *,
    quantization_enabled,
    device: str,
) -> str | None:
    """Return the in-process DiT quantization mode for the selected UI state."""
    return select_quantization_value(quantization_enabled, device=device)


def _resolve_project_root_for_generation() -> str:
    """Resolve the ACE-Step project root for auto-initialization."""

    project_root = os.environ.get("ACESTEP_PROJECT_ROOT")
    if project_root:
        return project_root
    return str(Path(__file__).resolve().parents[5])


def _dit_service_needs_reinit(
    dit_handler,
    *,
    config_path: str | None,
    device: str | None,
    use_flash_attention: bool,
    offload_to_cpu: bool,
    offload_dit_to_cpu: bool,
    compile_model: bool,
    quantization_enabled,
    mlx_dit: bool,
) -> tuple[bool, str | None]:
    """Return whether the foreground DiT handler must be reinitialized."""

    desired_quantization = _select_quantization_value(
        quantization_enabled=quantization_enabled,
        device=device or "auto",
    )

    if dit_handler is None:
        return False, desired_quantization

    if getattr(dit_handler, "model", None) is None:
        return True, desired_quantization

    last_init_params = getattr(dit_handler, "last_init_params", None) or {}
    if last_init_params.get("config_path") != config_path:
        return True, desired_quantization
    if device and device != "auto" and last_init_params.get("device") != device:
        return True, desired_quantization
    if bool(last_init_params.get("use_flash_attention")) != bool(use_flash_attention):
        return True, desired_quantization
    if bool(last_init_params.get("offload_to_cpu")) != bool(offload_to_cpu):
        return True, desired_quantization
    if bool(last_init_params.get("offload_dit_to_cpu")) != bool(offload_dit_to_cpu):
        return True, desired_quantization
    if bool(last_init_params.get("compile_model")) != bool(compile_model):
        return True, desired_quantization
    if last_init_params.get("quantization") != desired_quantization:
        return True, desired_quantization
    if bool(last_init_params.get("use_mlx_dit", True)) != bool(mlx_dit):
        return True, desired_quantization
    return False, desired_quantization


def _lm_service_needs_init(
    llm_handler,
    *,
    init_llm_checkbox: bool,
    think_checkbox: bool,
    auto_score: bool,
    lm_model_path: str | None,
    backend_dropdown: str | None,
    device: str | None,
    offload_to_cpu: bool,
) -> tuple[bool, str]:
    """Return whether the foreground LM handler must be initialized."""

    requested_lm_model = str(lm_model_path or "").strip() or _default_lm_model(
        llm_handler
    )
    needs_lm = bool(init_llm_checkbox or think_checkbox or auto_score)
    if not needs_lm or llm_handler is None:
        return False, requested_lm_model

    if not getattr(llm_handler, "llm_initialized", False):
        return True, requested_lm_model

    last_init_params = getattr(llm_handler, "last_init_params", None) or {}
    if not last_init_params:
        return True, requested_lm_model
    if last_init_params.get("lm_model_path") != requested_lm_model:
        return True, requested_lm_model
    if last_init_params.get("backend") != backend_dropdown:
        return True, requested_lm_model
    if device and device != "auto" and last_init_params.get("device") != device:
        return True, requested_lm_model
    if bool(last_init_params.get("offload_to_cpu")) != bool(offload_to_cpu):
        return True, requested_lm_model
    return False, requested_lm_model


def _unload_lm_when_unused(
    llm_handler,
    *,
    init_llm_checkbox: bool,
    think_checkbox: bool,
    auto_score: bool,
) -> str:
    """Unload the foreground LM when the selected generation path will not use it."""

    needs_lm = bool(init_llm_checkbox or think_checkbox or auto_score)
    if needs_lm or llm_handler is None:
        return ""
    if not getattr(llm_handler, "llm_initialized", False):
        return ""
    unload = getattr(llm_handler, "unload", None)
    if not callable(unload):
        return ""
    logger.info(
        "[generate_with_batch_management] Unloading foreground 5Hz LM; selected run does not use LM."
    )
    unload()
    return "Unloaded 5Hz LM for direct DiT generation."


def _default_lm_model(llm_handler) -> str:
    """Return the current GPU-tier LM default for foreground generation."""
    fallback_model = "acestep-5Hz-lm-1.7B"
    try:
        from acestep.model_downloader import DEFAULT_LM_MODEL

        fallback_model = DEFAULT_LM_MODEL
    except Exception:
        pass

    available_models: list[str] = []
    if llm_handler is not None:
        try:
            available_models = list(llm_handler.get_available_5hz_lm_models())
        except Exception:
            available_models = []

    try:
        gpu_config = get_global_gpu_config()
        recommended_lm = getattr(gpu_config, "recommended_lm_model", "")
        selected_model = find_best_lm_model_on_disk(recommended_lm, available_models)
        return selected_model or recommended_lm or fallback_model
    except Exception:
        return (
            fallback_model
            if fallback_model in available_models
            else None
        ) or (available_models[0] if available_models else fallback_model)


def _ensure_in_process_service_ready(
    dit_handler,
    llm_handler,
    *,
    config_path: str | None,
    device: str | None,
    lm_model_path: str | None,
    backend_dropdown: str | None,
    init_llm_checkbox: bool,
    use_flash_attention_checkbox: bool,
    offload_to_cpu_checkbox: bool,
    offload_dit_to_cpu_checkbox: bool,
    compile_model_checkbox: bool,
    quantization_checkbox,
    mlx_dit_checkbox: bool,
    think_checkbox: bool,
    auto_score: bool,
) -> tuple[bool, str]:
    """Auto-initialize foreground handlers for Generate when required."""

    if dit_handler is None:
        return True, ""

    project_root = _resolve_project_root_for_generation()
    selected_model = str(config_path or "").strip() or DEFAULT_TURBO_DIT_MODEL
    status_lines: list[str] = []

    dit_requires_init, quant_value = _dit_service_needs_reinit(
        dit_handler,
        config_path=selected_model,
        device=device,
        use_flash_attention=bool(use_flash_attention_checkbox),
        offload_to_cpu=bool(offload_to_cpu_checkbox),
        offload_dit_to_cpu=bool(offload_dit_to_cpu_checkbox),
        compile_model=bool(compile_model_checkbox),
        quantization_enabled=quantization_checkbox,
        mlx_dit=bool(mlx_dit_checkbox),
    )
    if dit_requires_init:
        logger.info(
            "[generate_with_batch_management] Auto-initializing foreground DiT service: {}",
            selected_model,
        )
        status_lines.append(f"Initializing DiT service: {selected_model}")
        init_status, ok = dit_handler.initialize_service(
            project_root=project_root,
            config_path=selected_model,
            device=device or "auto",
            use_flash_attention=bool(use_flash_attention_checkbox),
            compile_model=bool(compile_model_checkbox),
            offload_to_cpu=bool(offload_to_cpu_checkbox),
            offload_dit_to_cpu=bool(offload_dit_to_cpu_checkbox),
            quantization=quant_value,
            use_mlx_dit=bool(mlx_dit_checkbox),
        )
        status_lines.append(init_status)
        check_generation_cancelled()
        if not ok:
            return False, "\n".join(status_lines)

    lm_unload_status = _unload_lm_when_unused(
        llm_handler,
        init_llm_checkbox=bool(init_llm_checkbox),
        think_checkbox=bool(think_checkbox),
        auto_score=bool(auto_score),
    )
    if lm_unload_status:
        status_lines.append(lm_unload_status)
    check_generation_cancelled()

    lm_requires_init, requested_lm_model = _lm_service_needs_init(
        llm_handler,
        init_llm_checkbox=bool(init_llm_checkbox),
        think_checkbox=bool(think_checkbox),
        auto_score=bool(auto_score),
        lm_model_path=lm_model_path,
        backend_dropdown=backend_dropdown,
        device=device,
        offload_to_cpu=bool(offload_to_cpu_checkbox),
    )
    if lm_requires_init:
        models_dir = Path(project_root) / "models"
        try:
            from acestep.model_downloader import get_models_dir

            models_dir = get_models_dir(project_root=project_root)
        except Exception:
            pass

        logger.info(
            "[generate_with_batch_management] Auto-initializing foreground 5Hz LM: {}",
            requested_lm_model,
        )
        status_lines.append(f"Initializing 5Hz LM: {requested_lm_model}")
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=str(models_dir),
            lm_model_path=requested_lm_model,
            backend=backend_dropdown or "pt",
            device=device or "auto",
            offload_to_cpu=bool(offload_to_cpu_checkbox),
            dtype=None,
        )
        llm_handler.last_init_params = {
            "lm_model_path": requested_lm_model,
            "backend": backend_dropdown,
            "device": device or "auto",
            "offload_to_cpu": bool(offload_to_cpu_checkbox),
        }
        status_lines.append(lm_status)
        check_generation_cancelled()
        if not lm_ok:
            return False, "\n".join(status_lines)

    return True, "\n".join(status_lines)


def _generate_with_batch_management_impl(
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
    track_name,
    complete_track_classes,
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
    autogen_checkbox,
    current_batch_index,
    total_batches,
    batch_queue,
    generation_params_state,
    subprocess_mode_checkbox,
    config_path,
    device,
    lm_model_path,
    backend_dropdown,
    init_llm_checkbox,
    use_flash_attention_checkbox,
    offload_to_cpu_checkbox,
    offload_dit_to_cpu_checkbox,
    compile_model_checkbox,
    quantization_checkbox,
    mlx_dit_checkbox,
    lora_dropdown,
    lora_path,
    use_lora_checkbox,
    lora_scale_slider,
    generate_lm_audio_codes=None,
    progress=gr.Progress(track_tqdm=True),
):
    """Wrap ``generate_with_progress`` with batch queue management state."""
    check_generation_cancelled()
    _ = (generation_params_state, use_lora_checkbox)  # reserved for API compatibility
    selected_model = str(config_path or "").strip() or DEFAULT_TURBO_DIT_MODEL
    logger.info(
        f"[generate_with_batch_management] Starting generation: "
        f"model={selected_model}, inference_steps={inference_steps}, "
        f"songs={normalize_generation_count(batch_size_input)}, duration={audio_duration}, "
        f"subprocess={bool(subprocess_mode_checkbox)}"
    )

    saved_params = _build_saved_params(
        captions, lyrics, bpm, key_scale, time_signature, vocal_language,
        inference_steps, guidance_scale, random_seed_checkbox, seed,
        reference_audio, audio_duration, batch_size_input, src_audio,
        text2music_audio_code_string, repainting_start, repainting_end,
        instruction_display_gen, audio_cover_strength, cover_noise_strength, task_type,
        no_fsq, use_adg, cfg_interval_start, cfg_interval_end, shift, infer_method,
        sampler_mode, velocity_norm_threshold, velocity_ema_factor,
        dcw_enabled, dcw_mode, dcw_scaler, dcw_high_scaler, dcw_wavelet,
        audio_format, mp3_bitrate, mp3_sample_rate, lm_temperature,
        think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
        use_cot_metas, use_cot_caption, use_cot_language,
        constrained_decoding_debug, allow_lm_batch, auto_score, auto_lrc,
        score_scale, lm_batch_chunk_size,
        track_name, complete_track_classes,
        enable_normalization, normalization_db, fade_in_duration, fade_out_duration,
        latent_shift, latent_rescale,
        repaint_mode=repaint_mode, repaint_strength=repaint_strength,
        retake_variance=retake_variance, retake_seed=retake_seed,
        flow_edit_morph=flow_edit_morph,
        flow_edit_source_caption=flow_edit_source_caption,
        flow_edit_source_lyrics=flow_edit_source_lyrics,
        flow_edit_n_min=flow_edit_n_min,
        flow_edit_n_max=flow_edit_n_max,
        flow_edit_n_avg=flow_edit_n_avg,
        lora_path=lora_path,
        lora_dropdown=lora_dropdown,
        lora_scale=lora_scale_slider,
        use_lora=bool(resolve_effective_lora_path(lora_path, lora_dropdown)),
        generate_lm_audio_codes=generate_lm_audio_codes,
    )
    saved_params["_subprocess_mode"] = bool(subprocess_mode_checkbox)

    if subprocess_mode_checkbox:
        effective_lora_path = resolve_effective_lora_path(lora_path, lora_dropdown)
        subprocess_generation_params = dict(saved_params)
        _apply_param_defaults(subprocess_generation_params)
        subprocess_generation_params["custom_timesteps"] = custom_timesteps or ""
        subprocess_generation_params["is_format_caption"] = bool(is_format_caption)

        project_root = os.environ.get("ACESTEP_PROJECT_ROOT")
        if not project_root:
            project_root = str(Path(__file__).resolve().parents[5])

        request_payload = {
            "project_root": project_root,
            "service": {
                "config_path": selected_model,
                "device": device,
                "lm_model_path": lm_model_path,
                "backend": backend_dropdown,
                "init_llm": init_llm_checkbox,
                "use_flash_attention": use_flash_attention_checkbox,
                "offload_to_cpu": offload_to_cpu_checkbox,
                "offload_dit_to_cpu": offload_dit_to_cpu_checkbox,
                "compile_model": compile_model_checkbox,
                "quantization": quantization_checkbox,
                "mlx_dit": mlx_dit_checkbox,
                "lora_path": effective_lora_path,
                "use_lora": bool(effective_lora_path),
                "lora_scale": lora_scale_slider,
            },
            "generation": {
                **subprocess_generation_params,
            },
        }
        active_output_dir = get_active_results_dir()
        if active_output_dir is not None:
            request_payload["output_dir"] = str(active_output_dir)
        active_run_name = get_active_generation_run_name()
        if active_run_name:
            request_payload["run_name"] = active_run_name

        subprocess_result = None
        try:
            for event in stream_subprocess_generation(request_payload):
                if event["kind"] == "status":
                    yield build_pending_core_outputs(
                        event["message"], is_format_caption
                    ) + (
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    )
                else:
                    subprocess_result = event["result"]
        except Exception as exc:
            error_msg = t("messages.batch_failed", error=str(exc))
            logger.exception("[generate_with_batch_management] Subprocess generation failed")
            gr.Warning(error_msg)
            yield build_pending_core_outputs(error_msg, is_format_caption) + (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
            return

        if subprocess_result is None:
            error_msg = t("messages.batch_failed", error="Subprocess produced no result")
            gr.Warning(error_msg)
            yield build_pending_core_outputs(error_msg, is_format_caption) + (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
            return

        all_audio_paths = subprocess_result.get("all_audio_paths")
        if all_audio_paths is None:
            yield build_pending_core_outputs(
                subprocess_result.get("status_output", "Generation failed"),
                is_format_caption,
            ) + (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
            return

        ui_core_list = list(build_final_core_outputs(subprocess_result))
        generation_info = subprocess_result.get("generation_info", "")
        seed_value_for_ui = subprocess_result.get("seed_value", "")
        lm_generated_metadata = subprocess_result.get("lm_metadata", {})
        scores_from_fg = list(subprocess_result.get("scores", []) or [])
        raw_codes_list = subprocess_result.get("codes", [""] * 8)
        generated_codes_batch = raw_codes_list if isinstance(raw_codes_list, list) else [""] * 8
        generated_codes_single = generated_codes_batch[0] if generated_codes_batch else ""
        extra_outputs_from_result = subprocess_result.get("extra_outputs", {}) or {}

        generation_count = normalize_generation_count(batch_size_input)
        if generation_count >= 2:
            codes_to_store = generated_codes_batch[:generation_count]
        else:
            codes_to_store = generated_codes_single

        batch_queue = store_batch_in_queue(
            batch_queue, current_batch_index,
            all_audio_paths, generation_info, seed_value_for_ui,
            scores=scores_from_fg,
            codes=codes_to_store,
            allow_lm_batch=allow_lm_batch,
            batch_size=generation_count,
            generation_params=saved_params,
            lm_generated_metadata=lm_generated_metadata,
            extra_outputs=extra_outputs_from_result,
            status="completed",
        )

        if auto_lrc and extra_outputs_from_result:
            batch_queue[current_batch_index]["lrcs"] = extra_outputs_from_result.get("lrcs", [""] * 8)
            batch_queue[current_batch_index]["subtitles"] = extra_outputs_from_result.get("subtitles", [None] * 8)

        total_batches = max(total_batches, current_batch_index + 1)
        batch_indicator_text = update_batch_indicator(current_batch_index, total_batches)
        can_prev, can_next = update_navigation_buttons(current_batch_index, total_batches)
        next_batch_status_text = t("messages.autogen_enabled") if autogen_checkbox else ""
        next_params = saved_params.copy()
        next_params["text2music_audio_code_string"] = ""
        next_params["random_seed_checkbox"] = True

        yield tuple(ui_core_list) + (
            current_batch_index, total_batches, batch_queue, next_params,
            batch_indicator_text,
            gr.update(interactive=can_prev),
            gr.update(interactive=can_next),
            next_batch_status_text,
            gr.update(interactive=True),
        )
        time_module.sleep(0.1)
        return

    gc.collect()
    torch = _get_torch_for_in_process()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()

    init_ready, init_message = _ensure_in_process_service_ready(
        dit_handler,
        llm_handler,
        config_path=selected_model,
        device=device,
        lm_model_path=lm_model_path,
        backend_dropdown=backend_dropdown,
        init_llm_checkbox=init_llm_checkbox,
        use_flash_attention_checkbox=use_flash_attention_checkbox,
        offload_to_cpu_checkbox=offload_to_cpu_checkbox,
        offload_dit_to_cpu_checkbox=offload_dit_to_cpu_checkbox,
        compile_model_checkbox=compile_model_checkbox,
        quantization_checkbox=quantization_checkbox,
        mlx_dit_checkbox=mlx_dit_checkbox,
        think_checkbox=think_checkbox,
        auto_score=auto_score,
    )
    if init_message:
        yield build_pending_core_outputs(init_message, is_format_caption) + (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        )
    check_generation_cancelled()
    if not init_ready:
        error_msg = t("messages.batch_failed", error=init_message or "Service initialization failed")
        logger.warning("[generate_with_batch_management] Foreground auto-init failed")
        gr.Warning(error_msg)
        yield build_pending_core_outputs(error_msg, is_format_caption) + (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        )
        return

    try:
        _resolved_lora_path, _use_lora, lora_status = apply_lora_selection_for_generation(
            dit_handler,
            lora_path,
            lora_dropdown,
            lora_scale_slider,
        )
    except Exception as exc:
        error_msg = t("messages.batch_failed", error=str(exc))
        logger.warning("[generate_with_batch_management] LoRA synchronization failed")
        gr.Warning(error_msg)
        yield build_pending_core_outputs(error_msg, is_format_caption) + (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        )
        return
    if lora_status:
        logger.info("[generate_with_batch_management] {}", lora_status)

    generator = generate_with_progress(
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
        allow_lm_batch, auto_score, auto_lrc, score_scale,
        lm_batch_chunk_size,
        enable_normalization, normalization_db, fade_in_duration, fade_out_duration,
        latent_shift, latent_rescale,
        repaint_mode, repaint_strength,
        retake_variance, retake_seed,
        flow_edit_morph, flow_edit_source_caption, flow_edit_source_lyrics,
        flow_edit_n_min, flow_edit_n_max, flow_edit_n_avg,
        generate_lm_audio_codes,
        progress,
    )

    final_result_from_inner = None
    for partial_result in generator:
        final_result_from_inner = partial_result
        if not IS_WINDOWS:
            ui_result = _extract_ui_core_outputs(partial_result)
            yield ui_result + (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )

    # Release the generator frame and run GC to reclaim any accelerator memory
    # that was not yet freed at the end of the inner generator.
    del generator
    gc.collect()
    torch = _get_torch_for_in_process()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()

    result = final_result_from_inner
    if result is None:
        error_msg = t("messages.batch_failed", error="No generation result was produced")
        logger.warning("[generate_with_batch_management] generate_with_progress yielded no results")
        gr.Warning(error_msg)
        yield (gr.skip(),) * 55
        return

    all_audio_paths = result[8]

    if all_audio_paths is None:
        ui_result = _extract_ui_core_outputs(result)
        yield ui_result + (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        )
        return

    generation_info = result[9]
    seed_value_for_ui = result[11]
    lm_generated_metadata = result[44]

    raw_codes_list = result[47] if len(result) > 47 else [""] * 8
    generated_codes_batch = raw_codes_list if isinstance(raw_codes_list, list) else [""] * 8
    generated_codes_single = generated_codes_batch[0] if generated_codes_batch else ""

    generation_count = normalize_generation_count(batch_size_input)
    if generation_count >= 2:
        codes_to_store = generated_codes_batch[:generation_count]
    else:
        codes_to_store = generated_codes_single

    next_params = saved_params.copy()
    next_params["text2music_audio_code_string"] = ""
    next_params["random_seed_checkbox"] = True

    extra_outputs_from_result = result[46] if len(result) > 46 and result[46] is not None else {}

    scores_from_fg = _extract_scores(result)

    batch_queue = store_batch_in_queue(
        batch_queue, current_batch_index,
        all_audio_paths, generation_info, seed_value_for_ui,
        scores=scores_from_fg,
        codes=codes_to_store,
        allow_lm_batch=allow_lm_batch,
        batch_size=generation_count,
        generation_params=saved_params,
        lm_generated_metadata=lm_generated_metadata,
        extra_outputs=extra_outputs_from_result,
        status="completed",
    )

    if auto_lrc and extra_outputs_from_result:
        batch_queue[current_batch_index]["lrcs"] = extra_outputs_from_result.get("lrcs", [""] * 8)
        batch_queue[current_batch_index]["subtitles"] = extra_outputs_from_result.get("subtitles", [None] * 8)

    total_batches = max(total_batches, current_batch_index + 1)
    batch_indicator_text = update_batch_indicator(current_batch_index, total_batches)
    can_prev, can_next = update_navigation_buttons(current_batch_index, total_batches)
    next_batch_status_text = t("messages.autogen_enabled") if autogen_checkbox else ""

    ui_core_list = list(_extract_ui_core_outputs(result))

    if auto_lrc and isinstance(extra_outputs_from_result, dict):
        lrcs = extra_outputs_from_result.get("lrcs", [""] * 8)
        for i in range(min(8, len(lrcs))):
            if lrcs[i]:
                ui_core_list[36 + i] = gr.update(value=lrcs[i], visible=True)

    logger.info(f"[generate_with_batch_management] Final yield: {len(ui_core_list)} core + 9 state")

    yield tuple(ui_core_list) + (
        current_batch_index, total_batches, batch_queue, next_params,
        batch_indicator_text,
        gr.update(interactive=can_prev),
        gr.update(interactive=can_next),
        next_batch_status_text,
        gr.update(interactive=True),
    )
    time_module.sleep(0.1)


def generate_with_batch_management(*args, **kwargs):
    """Run batch-managed generation under a cancellation scope."""

    is_format_caption = _extract_is_format_caption_arg(args, kwargs)
    with generation_cancel_scope():
        try:
            check_generation_cancelled()
            yield from _generate_with_batch_management_impl(*args, **kwargs)
        except GenerationCancelled:
            logger.info("[generate_with_batch_management] {}", CANCEL_MESSAGE)
            cleanup_runtime_memory()
            yield _cancelled_generation_outputs(is_format_caption)
        finally:
            if is_generation_cancelled():
                cleanup_runtime_memory()


def _cancelled_generation_outputs(is_format_caption: bool) -> tuple:
    """Return a 55-output Gradio update tuple for cancelled generation."""

    return build_pending_core_outputs(CANCEL_MESSAGE, is_format_caption) + (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        CANCEL_MESSAGE,
        gr.update(interactive=False),
    )


def _extract_is_format_caption_arg(args: tuple, kwargs: dict) -> bool:
    """Extract the ``is_format_caption`` positional argument for cancel output."""

    if "is_format_caption" in kwargs:
        return bool(kwargs["is_format_caption"])
    # Positional index 50 follows dit_handler, llm_handler, and generation inputs.
    return bool(args[50]) if len(args) > 50 else False


generate_with_batch_management.__signature__ = inspect.signature(_generate_with_batch_management_impl)
