"""CLI entry point for LoRA checkpoint sample generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from acestep.training.sample_generation import resolve_checkpoint_adapter_path


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Run sample generation inside the child process."""

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=args.project_root,
        config_path=args.config_path,
        device=args.device,
        offload_to_cpu=bool(args.offload_generation),
        quantization=None,
    )
    if not ok:
        return {"success": False, "error": status, "audios": [], "peak_vram_gb": _peak_vram()}

    adapter_path = resolve_checkpoint_adapter_path(args.checkpoint_dir)
    load_status = handler.load_lora(adapter_path)
    if "loaded" not in load_status.lower():
        return {
            "success": False,
            "error": load_status,
            "audios": [],
            "peak_vram_gb": _peak_vram(),
        }
    handler.set_lora_scale(1.0)
    handler.set_use_lora(True)

    params = GenerationParams(
        task_type="text2music",
        thinking=False,
        caption=args.prompt,
        lyrics=args.lyrics,
        instrumental="[instrumental]" in str(args.lyrics).lower(),
        vocal_language="en",
        duration=float(args.duration),
        inference_steps=int(args.inference_steps),
        guidance_scale=1.0,
        seed=int(args.seed),
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[int(args.seed)],
        audio_format="flac",
    )
    result = generate_music(handler, None, params=params, config=config, save_dir=args.output_dir)
    return {
        "success": bool(result.success),
        "error": result.error or result.status_message,
        "audios": _serializable_audios(result.audios),
        "peak_vram_gb": _peak_vram(),
    }


def _serializable_audios(audios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return sample audio metadata safe for JSON output."""

    serializable: list[dict[str, Any]] = []
    for audio in audios:
        serializable.append(
            {
                "path": str(audio.get("path") or ""),
                "key": str(audio.get("key") or ""),
                "sample_rate": int(audio.get("sample_rate") or 0),
            }
        )
    return serializable


def _peak_vram() -> float:
    """Return child-process CUDA peak allocation in GiB."""

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for child sample generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--offload-generation", action="store_true")
    return parser


def main() -> None:
    """CLI entry point used by the training sample subprocess."""

    args = _build_parser().parse_args()
    payload = _run_cli(args)
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload.get("success"):
        logger.error("Sample generation failed: {}", payload.get("error"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
