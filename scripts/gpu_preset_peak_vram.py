"""Measure peak CUDA VRAM for ACE-Step GPU presets.

This is a local verification helper for the premium Gradio GPU preset matrix.
It runs one fresh Python process per case so CUDA allocator state does not leak
between tiers or model variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DEFAULTS = {
    "acestep-v15-xl-sft": {
        "inference_steps": 50,
        "guidance_scale": 7.0,
        "use_adg": False,
    },
    "acestep-v15-xl-turbo": {
        "inference_steps": 8,
        "guidance_scale": 1.0,
        "use_adg": False,
    },
}
DEFAULT_TIERS = (
    "tier1",
    "tier2",
    "tier3",
    "tier4",
    "tier5",
    "tier6a",
    "tier6b",
    "unlimited",
)


WORKER = r"""
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

project_root = Path(os.environ["ACE_BENCH_PROJECT_ROOT"]).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from acestep.gpu_config import GPU_TIER_CONFIGS, resolve_lm_backend, set_global_gpu_config
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.handler import AceStepHandler
from acestep.lazy_runtime import build_startup_gpu_config_for_tier
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import DEFAULT_LM_MODEL, get_models_dir

model = os.environ["ACE_BENCH_MODEL"]
tier = os.environ["ACE_BENCH_TIER"]
duration = float(os.environ["ACE_BENCH_DURATION"])
batch_size = int(os.environ["ACE_BENCH_BATCH"])
use_lm = os.environ["ACE_BENCH_USE_LM"] == "1"
steps = int(os.environ["ACE_BENCH_STEPS"])
guidance_scale = float(os.environ["ACE_BENCH_GUIDANCE_SCALE"])
use_adg = os.environ["ACE_BENCH_USE_ADG"] == "1"

started = time.perf_counter()
result = {
    "model": model,
    "tier": tier,
    "duration": duration,
    "batch_size": batch_size,
    "use_lm": use_lm,
    "inference_steps": steps,
    "guidance_scale": guidance_scale,
    "use_adg": use_adg,
    "success": False,
    "init_success": False,
    "gen_success": False,
    "error": None,
    "wall_time_sec": 0.0,
    "peak_allocated_gib": 0.0,
    "peak_reserved_gib": 0.0,
    "memory_allocated_after_gib": 0.0,
    "memory_reserved_after_gib": 0.0,
}

save_dir = tempfile.mkdtemp(prefix=f"ace_gpu_preset_{tier}_{model}_")
dit_handler = None
llm_handler = None
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gpu_config = build_startup_gpu_config_for_tier(tier)
    set_global_gpu_config(gpu_config)
    result.update(
        {
            "offload_to_cpu": gpu_config.offload_to_cpu_default,
            "offload_dit_to_cpu": gpu_config.offload_dit_to_cpu_default,
            "quantization": "int8_weight_only" if gpu_config.quantization_default else None,
            "compile_model": True if gpu_config.quantization_default else gpu_config.compile_model_default,
            "init_lm_default": gpu_config.init_lm_default,
            "recommended_backend": gpu_config.recommended_backend,
            "recommended_lm_model": gpu_config.recommended_lm_model,
            "max_batch_with_lm": gpu_config.max_batch_size_with_lm,
            "max_batch_without_lm": gpu_config.max_batch_size_without_lm,
            "max_duration_with_lm": gpu_config.max_duration_with_lm,
            "max_duration_without_lm": gpu_config.max_duration_without_lm,
        }
    )

    dit_handler = AceStepHandler()
    status, ok = dit_handler.initialize_service(
        project_root=str(project_root),
        config_path=model,
        device="cuda",
        use_flash_attention=False,
        compile_model=result["compile_model"],
        offload_to_cpu=result["offload_to_cpu"],
        offload_dit_to_cpu=result["offload_dit_to_cpu"],
        quantization=result["quantization"],
    )
    result["dit_init_status"] = status
    if not ok:
        raise RuntimeError(f"DiT init failed: {status}")

    llm_handler = LLMHandler()
    if use_lm:
        lm_model = gpu_config.recommended_lm_model or DEFAULT_LM_MODEL
        models_dir = get_models_dir(project_root=project_root)
        if not (models_dir / lm_model).exists():
            lm_model = DEFAULT_LM_MODEL
        backend = resolve_lm_backend(gpu_config.recommended_backend, gpu_config)
        result["lm_model"] = lm_model
        result["lm_backend"] = backend
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=str(models_dir),
            lm_model_path=lm_model,
            backend=backend,
            device="cuda",
            offload_to_cpu=result["offload_to_cpu"],
            dtype=None,
        )
        result["lm_init_status"] = lm_status
        if not lm_ok:
            raise RuntimeError(f"LM init failed: {lm_status}")
    else:
        result["lm_model"] = None
        result["lm_backend"] = None

    result["init_success"] = True

    params = GenerationParams(
        caption=(
            "modern cinematic pop rap, warm synth pads, punchy drums, "
            "deep bass, clean confident vocal"
        ),
        lyrics=(
            "[verse]\n"
            "We build the light from a quiet spark\n"
            "Keep steady hands when the road gets dark\n"
            "[chorus]\n"
            "Rise up slow with a steady heart\n"
            "Let the whole room feel the start"
        ),
        vocal_language="en",
        duration=duration,
        thinking=use_lm,
        use_cot_metas=use_lm,
        use_cot_caption=False,
        use_cot_language=False,
        use_constrained_decoding=True,
        inference_steps=steps,
        guidance_scale=guidance_scale,
        use_adg=use_adg,
        seed=42,
        task_type="text2music",
    )
    config = GenerationConfig(
        batch_size=batch_size,
        seeds=[42 + idx for idx in range(batch_size)],
        use_random_seed=False,
        audio_format="flac",
    )
    generated = generate_music(dit_handler, llm_handler, params, config, save_dir=save_dir)
    result["gen_success"] = bool(generated.success)
    if not generated.success:
        raise RuntimeError(generated.error)

    result["success"] = True
    result["audio_count"] = len(generated.audios)
    result["time_costs"] = generated.extra_outputs.get("time_costs", {})
except torch.cuda.OutOfMemoryError as exc:
    result["error"] = f"CUDA OOM: {exc}"
except Exception as exc:
    result["error"] = str(exc)
finally:
    result["wall_time_sec"] = time.perf_counter() - started
    if torch.cuda.is_available():
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024 ** 3)
        result["memory_allocated_after_gib"] = torch.cuda.memory_allocated() / (1024 ** 3)
        result["memory_reserved_after_gib"] = torch.cuda.memory_reserved() / (1024 ** 3)
    try:
        if dit_handler is not None:
            dit_handler.model = None
            if hasattr(dit_handler, "vae"):
                dit_handler.vae = None
            if hasattr(dit_handler, "text_encoder"):
                dit_handler.text_encoder = None
        if llm_handler is not None and hasattr(llm_handler, "llm"):
            llm_handler.llm = None
    except Exception:
        pass
    shutil.rmtree(save_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("ACE_BENCH_JSON_START")
    print(json.dumps(result, sort_keys=True))
    print("ACE_BENCH_JSON_END")
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_DEFAULTS))
    parser.add_argument("--tiers", nargs="+", default=list(DEFAULT_TIERS))
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mode", choices=["ui-default", "max-batch"], default="max-batch")
    parser.add_argument("--include-lm", action="store_true")
    parser.add_argument("--lm-only", action="store_true")
    parser.add_argument("--output", default="ui_verification/gpu_preset_peak_vram.json")
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser.parse_args()


def _extract_json(stdout: str) -> dict:
    start = stdout.rfind("ACE_BENCH_JSON_START")
    end = stdout.rfind("ACE_BENCH_JSON_END")
    if start == -1 or end == -1 or end <= start:
        return {"success": False, "error": "Worker did not emit result JSON", "raw_stdout": stdout[-4000:]}
    payload = stdout[start + len("ACE_BENCH_JSON_START"):end].strip()
    return json.loads(payload)


def _batch_for_mode(tier: str, *, use_lm: bool, mode: str) -> int:
    from acestep.lazy_runtime import build_startup_gpu_config_for_tier

    cfg = build_startup_gpu_config_for_tier(tier)
    max_batch = cfg.max_batch_size_with_lm if use_lm else cfg.max_batch_size_without_lm
    return max(1, max_batch if mode == "max-batch" else min(2, max_batch))


def main() -> int:
    args = _parse_args()
    results: list[dict] = []
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        defaults = MODEL_DEFAULTS[model]
        for tier in args.tiers:
            use_lm_values = [] if args.lm_only else [False]
            if args.include_lm or args.lm_only:
                from acestep.lazy_runtime import build_startup_gpu_config_for_tier

                cfg = build_startup_gpu_config_for_tier(tier)
                if cfg.init_lm_default and cfg.recommended_lm_model:
                    use_lm_values.append(True)
            for use_lm in use_lm_values:
                batch = _batch_for_mode(tier, use_lm=use_lm, mode=args.mode)
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                        "ACESTEP_DISABLE_TQDM": "1",
                        "HF_HOME": str(PROJECT_ROOT / ".cache" / "huggingface"),
                        "HF_MODULES_CACHE": str(
                            PROJECT_ROOT / ".cache" / "huggingface" / "modules"
                        ),
                        "TRANSFORMERS_CACHE": str(
                            PROJECT_ROOT / ".cache" / "huggingface" / "transformers"
                        ),
                        "ACE_BENCH_MODEL": model,
                        "ACE_BENCH_PROJECT_ROOT": str(PROJECT_ROOT),
                        "ACE_BENCH_TIER": tier,
                        "ACE_BENCH_DURATION": str(args.duration),
                        "ACE_BENCH_BATCH": str(batch),
                        "ACE_BENCH_USE_LM": "1" if use_lm else "0",
                        "ACE_BENCH_STEPS": str(defaults["inference_steps"]),
                        "ACE_BENCH_GUIDANCE_SCALE": str(defaults["guidance_scale"]),
                        "ACE_BENCH_USE_ADG": "1" if defaults["use_adg"] else "0",
                    }
                )
                label = f"{model} {tier} {'LM' if use_lm else 'DiT'} batch={batch}"
                print(f"\n=== {label} ===", flush=True)
                started = time.perf_counter()
                completed = subprocess.run(
                    [str(PYTHON), "-c", WORKER],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                result = _extract_json(completed.stdout)
                result["returncode"] = completed.returncode
                result["stderr_tail"] = completed.stderr[-4000:]
                result["stdout_tail"] = completed.stdout[-4000:]
                result["case_wall_time_sec"] = time.perf_counter() - started
                results.append(result)
                output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                status = "PASS" if result.get("success") else "FAIL"
                print(
                    f"{status}: peak_alloc={result.get('peak_allocated_gib', 0):.2f} GiB, "
                    f"peak_reserved={result.get('peak_reserved_gib', 0):.2f} GiB, "
                    f"wall={result.get('wall_time_sec', 0):.1f}s",
                    flush=True,
                )
                if not result.get("success"):
                    print(f"error: {result.get('error')}", flush=True)

    print(f"\nWrote {len(results)} results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
