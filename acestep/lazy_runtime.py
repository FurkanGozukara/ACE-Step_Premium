"""Startup-safe runtime helpers for delayed initialization flows.

This module keeps the parent Gradio process free of heavyweight runtime imports
until an in-process action actually needs them.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from acestep.gpu_config import (
    DEBUG_MAX_CUDA_VRAM_ENV,
    DEBUG_MAX_MPS_VRAM_ENV,
    DEBUG_MAX_XPU_VRAM_ENV,
    GPUConfig,
    GPU_TIER_CONFIGS,
    SAVE_MEMORY_ENV,
    get_gpu_tier,
)
from acestep.model_downloader import DEFAULT_LM_MODEL, get_models_dir


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_CUDA_FLASH_NAME_PATTERNS = (
    r"\brtx\s*30",
    r"\brtx\s*40",
    r"\brtx\s*50",
    r"\ba10\b",
    r"\ba16\b",
    r"\ba30\b",
    r"\ba40\b",
    r"\ba100\b",
    r"\bal40\b",
    r"\bl4\b",
    r"\bh100\b",
    r"\bh200\b",
)


@dataclass(frozen=True)
class StartupGpuState:
    """Lightweight accelerator metadata collected without importing torch."""

    gpu_config: GPUConfig
    device_name: str
    device_kind: str
    cuda_compute_major: int | None = None


def _env_truthy(name: str) -> bool:
    """Return whether the named environment variable is set to a true value."""

    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _read_debug_memory_gb(name: str) -> float | None:
    """Return a debug override VRAM value in GB when present and valid."""

    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid {} value: {}", name, raw)
        return None
    return max(value, 0.0)


def _run_command(*args: str, timeout: int = 5) -> str | None:
    """Run a subprocess command and return stdout on success."""

    try:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or "").strip()
    return text or None


def _heuristic_cuda_compute_major(device_name: str) -> int | None:
    """Infer a likely CUDA compute-major version from the GPU name."""

    lowered = (device_name or "").lower()
    if any(re.search(pattern, lowered) for pattern in _CUDA_FLASH_NAME_PATTERNS):
        return 8
    if re.search(r"\brtx\s*20", lowered) or any(
        token in lowered for token in ("t4", "v100", "titan v")
    ):
        return 7
    return None


def _detect_nvidia_gpu() -> tuple[float, str, int | None] | None:
    """Probe NVIDIA GPU info without importing torch."""

    debug_memory = _read_debug_memory_gb(DEBUG_MAX_CUDA_VRAM_ENV)
    if debug_memory is not None:
        device_name = (
            _run_command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
            or "NVIDIA GPU"
        )
        return debug_memory, device_name.splitlines()[0].strip(), _heuristic_cuda_compute_major(device_name)

    gpu_rows = _run_command(
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    )
    if not gpu_rows:
        return None

    first_row = gpu_rows.splitlines()[0]
    parts = [part.strip() for part in first_row.split(",")]
    if len(parts) < 2:
        return None

    device_name = parts[0] or "NVIDIA GPU"
    try:
        total_memory_mib = float(parts[1])
    except ValueError:
        return None

    compute_major = None
    compute_text = _run_command(
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader,nounits",
    )
    if compute_text:
        try:
            compute_major = int(float(compute_text.splitlines()[0].strip()))
        except ValueError:
            compute_major = None
    if compute_major is None:
        compute_major = _heuristic_cuda_compute_major(device_name)

    return total_memory_mib / 1024.0, device_name, compute_major


def _detect_xpu_gpu() -> tuple[float, str] | None:
    """Probe Intel XPU info without importing torch."""

    debug_memory = _read_debug_memory_gb(DEBUG_MAX_XPU_VRAM_ENV)
    if debug_memory is not None:
        return debug_memory, "Intel XPU"

    text = _run_command("xpu-smi", "discovery", "-j")
    if not text:
        return None

    match_name = re.search(r'"device_name"\s*:\s*"([^"]+)"', text)
    name = match_name.group(1).strip() if match_name else "Intel XPU"

    memory_matches = re.findall(
        r'"memory_physical_size_byte"\s*:\s*"?(\d+)"?',
        text,
    )
    if not memory_matches:
        return None

    try:
        memory_bytes = max(int(value) for value in memory_matches)
    except ValueError:
        return None
    return memory_bytes / (1024.0**3), name


def _detect_apple_silicon() -> tuple[float, str] | None:
    """Probe Apple Silicon unified memory without importing torch."""

    debug_memory = _read_debug_memory_gb(DEBUG_MAX_MPS_VRAM_ENV)
    if debug_memory is not None:
        return debug_memory, "Apple Silicon (MPS)"

    if sys.platform != "darwin":
        return None

    mem_text = _run_command("sysctl", "-n", "hw.memsize")
    if not mem_text:
        return None

    try:
        total_bytes = int(mem_text.splitlines()[0].strip())
    except ValueError:
        return None

    chip = _run_command("sysctl", "-n", "machdep.cpu.brand_string") or "Apple Silicon"
    return (total_bytes / (1024.0**3)) * 0.75, f"{chip.strip()} (MPS)"


def _build_gpu_config_for_tier(
    tier: str,
    gpu_memory_gb: float,
    device_kind: str,
) -> GPUConfig:
    """Construct a startup-safe GPUConfig without runtime torch probes."""

    config = GPU_TIER_CONFIGS[tier]

    lm_backend_restriction = config.get("lm_backend_restriction", "all")
    recommended_backend = config.get("recommended_backend", "vllm")
    offload_to_cpu_default = config.get("offload_to_cpu_default", True)
    offload_dit_to_cpu_default = config.get("offload_dit_to_cpu_default", True)
    quantization_default = config.get("quantization_default", True)
    compile_model_default = config.get("compile_model_default", True)

    if device_kind == "mps":
        lm_backend_restriction = "pt_mlx_only"
        recommended_backend = "mlx"
        offload_to_cpu_default = False
        offload_dit_to_cpu_default = False
        quantization_default = False
        compile_model_default = False
    elif device_kind in {"cpu", "xpu"}:
        lm_backend_restriction = "pt_only"
        recommended_backend = "pt"
    elif device_kind == "cuda":
        try:
            from acestep.llm_backend_compat import get_vllm_preflight_warning

            if get_vllm_preflight_warning(device="cuda") is not None:
                recommended_backend = "pt"
        except Exception:
            pass

    gpu_config = GPUConfig(
        tier=tier,
        gpu_memory_gb=gpu_memory_gb,
        max_duration_with_lm=config["max_duration_with_lm"],
        max_duration_without_lm=config["max_duration_without_lm"],
        max_batch_size_with_lm=config["max_batch_size_with_lm"],
        max_batch_size_without_lm=config["max_batch_size_without_lm"],
        init_lm_default=config["init_lm_default"],
        available_lm_models=config["available_lm_models"],
        recommended_lm_model=config.get("recommended_lm_model", ""),
        lm_backend_restriction=lm_backend_restriction,
        recommended_backend=recommended_backend,
        offload_to_cpu_default=offload_to_cpu_default,
        offload_dit_to_cpu_default=offload_dit_to_cpu_default,
        quantization_default=quantization_default,
        compile_model_default=compile_model_default,
        lm_memory_gb=config["lm_memory_gb"],
        save_memory_mode=_env_truthy(SAVE_MEMORY_ENV),
        mlx_vae_chunk_size=512,
    )
    return gpu_config


def _build_gpu_config(gpu_memory_gb: float, device_kind: str) -> GPUConfig:
    """Construct a startup-safe GPUConfig from detected memory size."""

    return _build_gpu_config_for_tier(get_gpu_tier(gpu_memory_gb), gpu_memory_gb, device_kind)


def build_startup_gpu_state() -> StartupGpuState:
    """Return startup GPU metadata without importing torch into the parent app."""

    nvidia_state = _detect_nvidia_gpu()
    if nvidia_state is not None:
        memory_gb, device_name, compute_major = nvidia_state
        return StartupGpuState(
            gpu_config=_build_gpu_config(memory_gb, "cuda"),
            device_name=device_name,
            device_kind="cuda",
            cuda_compute_major=compute_major,
        )

    xpu_state = _detect_xpu_gpu()
    if xpu_state is not None:
        memory_gb, device_name = xpu_state
        return StartupGpuState(
            gpu_config=_build_gpu_config(memory_gb, "xpu"),
            device_name=device_name,
            device_kind="xpu",
        )

    apple_state = _detect_apple_silicon()
    if apple_state is not None:
        memory_gb, device_name = apple_state
        return StartupGpuState(
            gpu_config=_build_gpu_config(memory_gb, "mps"),
            device_name=device_name,
            device_kind="mps",
        )

    return StartupGpuState(
        gpu_config=_build_gpu_config(0.0, "cpu"),
        device_name="CPU only",
        device_kind="cpu",
    )


def build_startup_gpu_config_for_tier(
    tier: str,
    startup_gpu_state: StartupGpuState | None = None,
) -> GPUConfig:
    """Return a startup-safe tier override config without importing torch."""

    state = startup_gpu_state or build_startup_gpu_state()
    if tier not in GPU_TIER_CONFIGS:
        return state.gpu_config
    return _build_gpu_config_for_tier(
        tier=tier,
        gpu_memory_gb=state.gpu_config.gpu_memory_gb,
        device_kind=state.device_kind,
    )


def is_startup_flash_attention_available(
    device: str | None,
    startup_gpu_state: StartupGpuState | None,
) -> bool:
    """Estimate flash-attention availability without importing torch."""

    target_device = str(device or "auto").split(":", 1)[0]
    if target_device not in {"auto", "cuda"}:
        return False
    if startup_gpu_state is None or startup_gpu_state.device_kind != "cuda":
        return False
    if importlib.util.find_spec("flash_attn") is None:
        return False
    compute_major = startup_gpu_state.cuda_compute_major
    return compute_major is not None and compute_major >= 8


class LazyAceStepHandler:
    """Delay importing the heavy DiT runtime until an in-process action needs it."""

    def __init__(self, startup_gpu_state: StartupGpuState | None = None):
        object.__setattr__(self, "_startup_gpu_state", startup_gpu_state)
        object.__setattr__(self, "_real_handler", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_pending_attrs", {})
        object.__setattr__(self, "_last_init_params", None)

    def _ensure_real_handler(self):
        """Instantiate and return the real handler on first use."""

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None:
            return real_handler

        with object.__getattribute__(self, "_lock"):
            real_handler = object.__getattribute__(self, "_real_handler")
            if real_handler is None:
                from acestep.handler import AceStepHandler

                real_handler = AceStepHandler()
                pending_attrs = dict(object.__getattribute__(self, "_pending_attrs"))
                for key, value in pending_attrs.items():
                    try:
                        setattr(real_handler, key, value)
                    except Exception:
                        logger.debug(
                            "Skipping pending DiT attribute during lazy init: {}",
                            key,
                        )
                last_init_params = object.__getattribute__(self, "_last_init_params")
                if last_init_params is not None:
                    real_handler.last_init_params = last_init_params
                object.__setattr__(self, "_real_handler", real_handler)
        return real_handler

    def _models_dir(self, project_root: str | None = None) -> Path:
        """Resolve the local models directory."""

        return Path(get_models_dir(project_root=project_root))

    def _scan_model_dirs(self, prefix: str) -> list[str]:
        """Return sorted model subdirectories with the given prefix."""

        models_dir = self._models_dir()
        if not models_dir.exists():
            return []
        names = [
            child.name
            for child in models_dir.iterdir()
            if child.is_dir() and child.name.startswith(prefix)
        ]
        names.sort()
        return names

    def get_available_checkpoints(self) -> list[str]:
        """Return the local models directory as the available checkpoint root."""

        models_dir = self._models_dir()
        return [str(models_dir)] if models_dir.exists() else []

    def get_available_acestep_v15_models(self) -> list[str]:
        """Return locally available ACE-Step 1.5 DiT models."""

        return self._scan_model_dirs("acestep-v15-")

    def is_flash_attention_available(self, device: str | None = None) -> bool:
        """Estimate flash-attention support without importing torch."""

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None:
            return real_handler.is_flash_attention_available(device)
        return is_startup_flash_attention_available(
            device=device,
            startup_gpu_state=object.__getattribute__(self, "_startup_gpu_state"),
        )

    def is_turbo_model(self) -> bool:
        """Return whether the active model selection is a turbo variant."""

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None:
            return real_handler.is_turbo_model()
        last_init_params = object.__getattribute__(self, "_last_init_params") or {}
        config_path = str(last_init_params.get("config_path") or "").lower()
        return "turbo" in config_path

    def initialize_service(self, *args: Any, **kwargs: Any):
        """Initialize the real DiT runtime on demand."""

        handler = self._ensure_real_handler()
        status, enable = handler.initialize_service(*args, **kwargs)
        object.__setattr__(
            self,
            "_last_init_params",
            getattr(handler, "last_init_params", None),
        )
        return status, enable

    def load_lora(self, *args: Any, **kwargs: Any):
        """Lazy-load the runtime and forward LoRA loading."""

        return self._ensure_real_handler().load_lora(*args, **kwargs)

    def unload_lora(self, *args: Any, **kwargs: Any):
        """Lazy-load the runtime and forward LoRA unloading."""

        return self._ensure_real_handler().unload_lora(*args, **kwargs)

    def set_use_lora(self, *args: Any, **kwargs: Any):
        """Lazy-load the runtime and update LoRA activation state."""

        return self._ensure_real_handler().set_use_lora(*args, **kwargs)

    def set_lora_scale(self, *args: Any, **kwargs: Any):
        """Lazy-load the runtime and update LoRA scale."""

        return self._ensure_real_handler().set_lora_scale(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Expose selected placeholder state before the real handler exists."""

        if name == "model":
            return getattr(object.__getattribute__(self, "_real_handler"), "model", None)
        if name == "config":
            return getattr(object.__getattribute__(self, "_real_handler"), "config", None)
        if name == "device":
            return getattr(object.__getattribute__(self, "_real_handler"), "device", "cpu")
        if name == "last_init_params":
            real_handler = object.__getattribute__(self, "_real_handler")
            if real_handler is not None and hasattr(real_handler, "last_init_params"):
                return real_handler.last_init_params
            return object.__getattribute__(self, "_last_init_params")

        pending_attrs = object.__getattribute__(self, "_pending_attrs")
        if name in pending_attrs:
            return pending_attrs[name]

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None:
            return getattr(real_handler, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Store pending attributes until the real runtime exists."""

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name == "last_init_params":
            object.__setattr__(self, "_last_init_params", value)
            real_handler = object.__getattribute__(self, "_real_handler")
            if real_handler is not None:
                setattr(real_handler, name, value)
            return

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None and hasattr(real_handler, name):
            setattr(real_handler, name, value)
            return

        pending_attrs = object.__getattribute__(self, "_pending_attrs")
        pending_attrs[name] = value


class LazyLLMHandler:
    """Delay importing the heavy LLM runtime until it is actually needed."""

    def __init__(self):
        object.__setattr__(self, "_real_handler", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_pending_attrs", {})
        object.__setattr__(self, "_last_init_params", None)

    def _ensure_real_handler(self):
        """Instantiate and return the real LLM handler on first use."""

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None:
            return real_handler

        with object.__getattribute__(self, "_lock"):
            real_handler = object.__getattribute__(self, "_real_handler")
            if real_handler is None:
                from acestep.llm_inference import LLMHandler

                real_handler = LLMHandler()
                pending_attrs = dict(object.__getattribute__(self, "_pending_attrs"))
                for key, value in pending_attrs.items():
                    try:
                        setattr(real_handler, key, value)
                    except Exception:
                        logger.debug(
                            "Skipping pending LLM attribute during lazy init: {}",
                            key,
                        )
                last_init_params = object.__getattribute__(self, "_last_init_params")
                if last_init_params is not None:
                    real_handler.last_init_params = last_init_params
                object.__setattr__(self, "_real_handler", real_handler)
        return real_handler

    def _models_dir(self) -> Path:
        """Resolve the local models directory."""

        return Path(get_models_dir())

    def get_available_5hz_lm_models(self) -> list[str]:
        """Return locally available 5Hz LM models."""

        models_dir = self._models_dir()
        if not models_dir.exists():
            return []
        names = [
            child.name
            for child in models_dir.iterdir()
            if child.is_dir() and child.name.startswith("acestep-5Hz-lm-")
        ]
        names.sort()
        return names

    def initialize(self, *args: Any, **kwargs: Any):
        """Initialize the real LLM runtime on demand."""

        handler = self._ensure_real_handler()
        status, ok = handler.initialize(*args, **kwargs)
        object.__setattr__(
            self,
            "_last_init_params",
            getattr(handler, "last_init_params", None),
        )
        return status, ok

    def unload(self, *args: Any, **kwargs: Any):
        """Lazy-load the runtime and forward unload."""

        return self._ensure_real_handler().unload(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Expose selected placeholder state before the real handler exists."""

        if name == "llm_initialized":
            return getattr(
                object.__getattribute__(self, "_real_handler"),
                "llm_initialized",
                False,
            )
        if name == "last_init_params":
            real_handler = object.__getattribute__(self, "_real_handler")
            if real_handler is not None and hasattr(real_handler, "last_init_params"):
                return real_handler.last_init_params
            return object.__getattribute__(self, "_last_init_params")
        if name == "llm_backend":
            return getattr(object.__getattribute__(self, "_real_handler"), "llm_backend", None)
        if name == "device":
            return getattr(object.__getattribute__(self, "_real_handler"), "device", "cpu")

        pending_attrs = object.__getattribute__(self, "_pending_attrs")
        if name in pending_attrs:
            return pending_attrs[name]

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None:
            return getattr(real_handler, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Store pending attributes until the real runtime exists."""

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name == "last_init_params":
            object.__setattr__(self, "_last_init_params", value)
            real_handler = object.__getattribute__(self, "_real_handler")
            if real_handler is not None:
                setattr(real_handler, name, value)
            return

        real_handler = object.__getattribute__(self, "_real_handler")
        if real_handler is not None and hasattr(real_handler, name):
            setattr(real_handler, name, value)
            return

        pending_attrs = object.__getattribute__(self, "_pending_attrs")
        pending_attrs[name] = value

    def get_default_lm_model(self) -> str:
        """Return the preferred default LM model name."""

        available_models = self.get_available_5hz_lm_models()
        return (DEFAULT_LM_MODEL if DEFAULT_LM_MODEL in available_models else None) or (
            available_models[0] if available_models else DEFAULT_LM_MODEL
        )
