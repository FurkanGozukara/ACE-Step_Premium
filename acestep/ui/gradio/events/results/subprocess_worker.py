"""CLI worker for isolated subprocess-based ACE-Step generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import DEFAULT_LM_MODEL, get_models_dir
from acestep.ui.gradio.events.generation.quantization import select_quantization_value
from acestep.ui.gradio.events.results.batch_management_helpers import _extract_scores
from acestep.ui.gradio.events.results.generation_progress import generate_with_progress


def _coerce_quantization(selection: Any, device: str | None) -> str | None:
    return select_quantization_value(selection, device=device or "auto")


def _build_generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload["generation"]
    return {
        "captions": generation.get("captions"),
        "lyrics": generation.get("lyrics"),
        "bpm": generation.get("bpm"),
        "key_scale": generation.get("key_scale"),
        "time_signature": generation.get("time_signature"),
        "vocal_language": generation.get("vocal_language"),
        "inference_steps": generation.get("inference_steps"),
        "guidance_scale": generation.get("guidance_scale"),
        "random_seed_checkbox": generation.get("random_seed_checkbox"),
        "seed": generation.get("seed"),
        "reference_audio": generation.get("reference_audio"),
        "audio_duration": generation.get("audio_duration"),
        "batch_size_input": generation.get("batch_size_input"),
        "src_audio": generation.get("src_audio"),
        "text2music_audio_code_string": generation.get("text2music_audio_code_string"),
        "repainting_start": generation.get("repainting_start"),
        "repainting_end": generation.get("repainting_end"),
        "instruction_display_gen": generation.get("instruction_display_gen"),
        "audio_cover_strength": generation.get("audio_cover_strength"),
        "cover_noise_strength": generation.get("cover_noise_strength"),
        "task_type": generation.get("task_type"),
        "use_adg": generation.get("use_adg"),
        "cfg_interval_start": generation.get("cfg_interval_start"),
        "cfg_interval_end": generation.get("cfg_interval_end"),
        "shift": generation.get("shift"),
        "infer_method": generation.get("infer_method"),
        "sampler_mode": generation.get("sampler_mode"),
        "velocity_norm_threshold": generation.get("velocity_norm_threshold"),
        "velocity_ema_factor": generation.get("velocity_ema_factor"),
        "custom_timesteps": generation.get("custom_timesteps"),
        "audio_format": generation.get("audio_format"),
        "mp3_bitrate": generation.get("mp3_bitrate"),
        "mp3_sample_rate": generation.get("mp3_sample_rate"),
        "lm_temperature": generation.get("lm_temperature"),
        "think_checkbox": generation.get("think_checkbox"),
        "lm_cfg_scale": generation.get("lm_cfg_scale"),
        "lm_top_k": generation.get("lm_top_k"),
        "lm_top_p": generation.get("lm_top_p"),
        "lm_negative_prompt": generation.get("lm_negative_prompt"),
        "use_cot_metas": generation.get("use_cot_metas"),
        "use_cot_caption": generation.get("use_cot_caption"),
        "use_cot_language": generation.get("use_cot_language"),
        "is_format_caption": generation.get("is_format_caption", False),
        "constrained_decoding_debug": generation.get("constrained_decoding_debug"),
        "allow_lm_batch": generation.get("allow_lm_batch"),
        "auto_score": generation.get("auto_score"),
        "auto_lrc": generation.get("auto_lrc"),
        "score_scale": generation.get("score_scale"),
        "lm_batch_chunk_size": generation.get("lm_batch_chunk_size"),
        "enable_normalization": generation.get("enable_normalization"),
        "normalization_db": generation.get("normalization_db"),
        "fade_in_duration": generation.get("fade_in_duration"),
        "fade_out_duration": generation.get("fade_out_duration"),
        "latent_shift": generation.get("latent_shift"),
        "latent_rescale": generation.get("latent_rescale"),
        "repaint_mode": generation.get("repaint_mode"),
        "repaint_strength": generation.get("repaint_strength"),
    }


def _json_safe_extra(extra_outputs: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(extra_outputs, dict):
        return {}
    return {
        "lrcs": extra_outputs.get("lrcs", []),
        "subtitles": extra_outputs.get("subtitles", []),
    }


def _write_result(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="ACE-Step subprocess generation worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    try:
        with open(args.request, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        project_root = str(Path(payload["project_root"]).resolve())
        os.environ.setdefault("ACESTEP_PROJECT_ROOT", project_root)
        os.environ.setdefault(
            "ACESTEP_CHECKPOINTS_DIR",
            str(get_models_dir(project_root=project_root)),
        )

        service = payload["service"]
        generation = payload["generation"]
        selected_model = str(service.get("config_path") or "").strip() or "acestep-v15-xl-sft"
        requested_steps = generation.get("inference_steps")

        print(f"[Worker] Project root: {project_root}", flush=True)
        print(f"[Worker] Initializing DiT model: {selected_model}", flush=True)

        dit_handler = AceStepHandler()
        llm_handler = LLMHandler()

        init_status, ok = dit_handler.initialize_service(
            project_root=project_root,
            config_path=selected_model,
            device=service["device"],
            use_flash_attention=service["use_flash_attention"],
            compile_model=service["compile_model"],
            offload_to_cpu=service["offload_to_cpu"],
            offload_dit_to_cpu=service["offload_dit_to_cpu"],
            quantization=_coerce_quantization(
                service.get("quantization"),
                service.get("device"),
            ),
            use_mlx_dit=service.get("mlx_dit", True),
        )
        if not ok:
            raise RuntimeError(init_status)

        lora_path = str(service.get("lora_path") or "").strip()
        if lora_path:
            print(f"[Worker] Loading LoRA: {lora_path}", flush=True)
            load_status = dit_handler.load_lora(lora_path)
            load_status_l = str(load_status).lower()
            if any(marker in load_status_l for marker in ("failed", "invalid", "not found", "not initialized", "not supported")):
                raise RuntimeError(load_status)
            dit_handler.set_lora_scale(float(service.get("lora_scale", 1.0) or 1.0))
            dit_handler.set_use_lora(True)

        needs_lm = bool(
            service.get("init_llm")
            or generation.get("think_checkbox")
            or generation.get("auto_score")
        )
        if needs_lm:
            lm_model_path = str(service.get("lm_model_path") or DEFAULT_LM_MODEL).strip() or DEFAULT_LM_MODEL
            print(f"[Worker] Initializing LLM: {lm_model_path}", flush=True)
            lm_status, lm_ok = llm_handler.initialize(
                checkpoint_dir=str(get_models_dir(project_root=project_root)),
                lm_model_path=lm_model_path,
                backend=service.get("backend") or "pt",
                device=service.get("device") or "auto",
                offload_to_cpu=bool(service.get("offload_to_cpu")),
                dtype=None,
            )
            if not lm_ok:
                raise RuntimeError(lm_status)

        print(
            f"[Worker] Starting generation: model={selected_model}, "
            f"inference_steps={requested_steps}, "
            f"batch_size={generation.get('batch_size_input')}, "
            f"duration={generation.get('audio_duration')}",
            flush=True,
        )
        final_result = None
        for partial in generate_with_progress(
            dit_handler,
            llm_handler,
            **_build_generation_kwargs(payload),
        ):
            final_result = partial

        if final_result is None:
            raise RuntimeError("generate_with_progress yielded no final result")

        scores = _extract_scores(final_result)
        extra_outputs = final_result[46] if len(final_result) > 46 else {}
        result_payload = {
            "success": True,
            "all_audio_paths": final_result[8],
            "generation_info": final_result[9],
            "status_output": final_result[10],
            "seed_value": final_result[11],
            "scores": scores,
            "lrcs": list(final_result[36:44]),
            "lm_metadata": final_result[44],
            "is_format_caption": final_result[45],
            "extra_outputs": _json_safe_extra(extra_outputs),
            "codes": list(final_result[47]) if len(final_result) > 47 else [],
        }
        _write_result(args.result, result_payload)
        print("[Worker] Generation complete.", flush=True)
        return 0
    except Exception as exc:
        _write_result(
            args.result,
            {
                "success": False,
                "error": str(exc),
            },
        )
        print(f"[Worker] ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
