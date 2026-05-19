"""Checkpoint sample generation helpers for LoRA training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_sample_subprocess(
    *,
    project_root: str,
    config_path: str,
    device: str,
    checkpoint_dir: str,
    output_dir: str,
    prompt: str,
    lyrics: str,
    duration: float,
    inference_steps: int,
    seed: int,
    offload_generation: bool,
) -> dict[str, Any]:
    """Generate one checkpoint sample in an isolated Python process."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(output_dir) / "sample_result.json"
    cmd = [
        sys.executable,
        "-m",
        "acestep.training.sample_generation_cli",
        "--project-root",
        project_root,
        "--config-path",
        config_path,
        "--device",
        device,
        "--checkpoint-dir",
        checkpoint_dir,
        "--output-dir",
        output_dir,
        "--prompt",
        prompt,
        "--lyrics",
        lyrics,
        "--duration",
        str(float(duration)),
        "--inference-steps",
        str(int(inference_steps)),
        "--seed",
        str(int(seed)),
        "--result-path",
        str(result_path),
    ]
    if offload_generation:
        cmd.append("--offload-generation")

    completed = subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=60 * 60,
        check=False,
    )
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        payload = {"success": False, "audios": [], "peak_vram_gb": 0.0}
    payload["returncode"] = completed.returncode
    payload["stdout_tail"] = completed.stdout[-2000:]
    payload["stderr_tail"] = completed.stderr[-2000:]
    return payload


def resolve_checkpoint_adapter_path(checkpoint_dir: str | os.PathLike[str]) -> str:
    """Return the adapter directory stored inside a training checkpoint."""

    base = Path(checkpoint_dir)
    adapter = base / "adapter"
    if (adapter / "adapter_config.json").is_file():
        return str(adapter)
    if (base / "adapter_config.json").is_file():
        return str(base)
    return str(adapter if adapter.exists() else base)
