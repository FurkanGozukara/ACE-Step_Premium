"""CLI worker for isolated subprocess-based ACE-Step generation."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import (
    DEFAULT_LM_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
    get_models_dir,
)
from acestep.ui.gradio.events.generation.quantization import select_quantization_value
from acestep.ui.gradio.events.results.batch_management_helpers import (
    _apply_param_defaults,
    _extract_scores,
)
from acestep.ui.gradio.events.results.generation_progress import generate_with_progress
from acestep.ui.gradio.events.results.subprocess_worker_progress import worker_console_progress


_GENERATION_KWARG_EXCLUDES = {"dit_handler", "llm_handler", "progress"}

def _coerce_quantization(selection: Any, device: str | None) -> str | None:
    """Return the backend quantization mode for the worker's selected device."""
    return select_quantization_value(selection, device=device or "auto")


def _build_generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Return keyword arguments matching the current generation function."""
    generation = dict(payload["generation"])
    _apply_param_defaults(generation)
    generation.setdefault("is_format_caption", False)
    kwargs: dict[str, Any] = {}
    for name, parameter in inspect.signature(generate_with_progress).parameters.items():
        if name in _GENERATION_KWARG_EXCLUDES:
            continue
        if name in generation:
            kwargs[name] = generation[name]
            continue
        if parameter.default is inspect.Parameter.empty:
            raise RuntimeError(f"Subprocess payload missing generation parameter: {name}")
    return kwargs


def _json_safe_extra(extra_outputs: dict[str, Any] | None) -> dict[str, Any]:
    """Return JSON-serializable extra generation outputs."""
    if not isinstance(extra_outputs, dict):
        return {}
    return {
        "lrcs": extra_outputs.get("lrcs", []),
        "subtitles": extra_outputs.get("subtitles", []),
    }


def _write_result(path: str | Path, payload: dict[str, Any]) -> None:
    """Write the worker result JSON for the parent process."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    """Run one isolated ACE-Step generation request."""
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
        _apply_param_defaults(generation)
        selected_model = (
            str(service.get("config_path") or "").strip()
            or DEFAULT_TURBO_DIT_MODEL
        )
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
                compile_model=bool(service.get("compile_model")),
            )
            if not lm_ok:
                raise RuntimeError(lm_status)

        print(
            f"[Worker] Starting generation: model={selected_model}, "
            f"inference_steps={requested_steps}, "
            f"songs={generation.get('batch_size_input')}, "
            f"duration={generation.get('audio_duration')}",
            flush=True,
        )
        final_result = None
        for partial in generate_with_progress(
            dit_handler,
            llm_handler,
            progress=worker_console_progress,
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
